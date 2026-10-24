"""Marcos de coordenadas de ParaLingbot y sus conversiones (integración Stella-VSLAM, etapa 8).

Una sola fuente de verdad para las convenciones (docs/COORDINATE_FRAMES.md). Todo lo interno del
repo (LingBot, .npz, selector, PoseBuffer) usa la convención de cámara de OpenCV; lo que sale por
ROS2 usa REP-103 / REP-105. Este módulo hace la traducción y nada más.

Convenciones
  OpenCV / "CV" (cámara óptica): x derecha, y abajo, z adelante. Dextrógira.
  ROS (REP-103, cuerpo):         x adelante, y izquierda, z arriba. Dextrógira.
  Un vector en coordenadas CV se pasa a ROS con A_CV_TO_ROS:  v_ros = A · v_cv
      x_ros = z_cv,  y_ros = -x_cv,  z_ros = -y_cv
  (es la misma matriz que stella_vslam_ros llama rot_ros_to_cv_map_frame).

Marcos (todos en unidades del modelo, NO metros; ver la escala en COORDINATE_FRAMES.md)
  paralingbot_map             mundo de la sesión. Origen: centro óptico del primer frame. Ejes ROS
                              alineados con la cámara del primer frame (x = hacia donde miraba, z =
                              "arriba" de esa cámara). NO está nivelado con la gravedad.
  paralingbot_odom            igual a paralingbot_map (identidad estática): no hay odometría propia.
                              Existe para respetar la cadena REP-105 map -> odom -> cuerpo; en el robot
                              lo reemplaza el odom del robot.
  paralingbot_camera_link     la cámara con ejes ROS (cuerpo). Su pose en el mapa = pose de referencia.
  paralingbot_camera_optical  la cámara con ejes CV (óptico): frame_id de las imágenes publicadas.
  paralingbot_map_cv          el mundo interno del repo (ejes CV del primer frame). Es donde viven las
                              poses c2w de los .npz: T_map_cv<-optical = c2w.
  stella_map                  mundo de Stella (su nodo, map_frame renombrado), ejes ROS de la cámara con
                              la que Stella inicializó. Su relación con paralingbot_map es una Sim(3):
                              la parte rígida va por TF y la escala por un topic aparte (TF no admite
                              escala; regla del plan: no esconder la escala dentro de otra transformación).
"""
import numpy as np

MAP = "paralingbot_map"
ODOM = "paralingbot_odom"
CAMERA_LINK = "paralingbot_camera_link"
CAMERA_OPTICAL = "paralingbot_camera_optical"
MAP_CV = "paralingbot_map_cv"
STELLA_MAP = "stella_map"
STELLA_CAMERA_LINK = "stella_camera_link"

A_CV_TO_ROS = np.array([[0.0, 0.0, 1.0],
                        [-1.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0]])
A_ROS_TO_CV = A_CV_TO_ROS.T


def _h(R):
    T = np.eye(4)
    T[:3, :3] = R
    return T


# transformaciones fijas (4x4)
T_LINK_OPTICAL = _h(A_CV_TO_ROS)          # pose del marco óptico en camera_link: p_link = A · p_optical
T_MAP_MAPCV = _h(A_CV_TO_ROS)             # pose de map_cv en map: p_map = A · p_map_cv
# sus inversas, precalculadas (son rotaciones puras: inversa = transpuesta). Etapa 19: las conversiones
# se llaman por cada keyframe de cada mensaje de Stella y np.linalg.inv dominaba el hilo del puente.
T_OPTICAL_LINK = T_LINK_OPTICAL.T.copy()
T_MAPCV_MAP = T_MAP_MAPCV.T.copy()


def c2w_cv_to_map_link(c2w_cv):
    """Pose c2w del repo (óptico en map_cv) -> pose de camera_link en paralingbot_map (ROS).
    T_map<-link = T_map<-mapcv · T_mapcv<-optical · T_optical<-link = A · c2w · A^T."""
    return T_MAP_MAPCV @ np.asarray(c2w_cv, np.float64) @ T_OPTICAL_LINK


def map_link_to_c2w_cv(T_map_link):
    """Inversa de c2w_cv_to_map_link."""
    return T_MAPCV_MAP @ np.asarray(T_map_link, np.float64) @ T_LINK_OPTICAL


def map_optical(c2w_cv):
    """Pose del marco óptico de la cámara en paralingbot_map: T_map<-optical = A · c2w."""
    return T_MAP_MAPCV @ np.asarray(c2w_cv, np.float64)


def sim3_cv_to_ros(s, R, t):
    """Sim(3) entre dos mundos CV (p_a = s R p_b + t) -> la misma entre sus versiones ROS."""
    return s, A_CV_TO_ROS @ R @ A_ROS_TO_CV, A_CV_TO_ROS @ np.asarray(t, np.float64)


def mat_to_quat_xyzw(R):
    """Matriz de rotación -> cuaternión (x, y, z, w) con w >= 0 (para geometry_msgs)."""
    from pose_buffer import mat_to_quat
    return mat_to_quat(R)


def gravity_tilt_deg(c2w_cv_list):
    """Inclinación (grados) entre el "abajo" medio de las cámaras (eje +y óptico, llevado al mundo)
    y el -z de paralingbot_map. Es cuánto difiere el mapa de estar nivelado si quien sostenía la
    cámara la llevó, en promedio, derecha. Diagnóstico, no una corrección."""
    down_cv = np.mean([np.asarray(T)[:3, 1] for T in c2w_cv_list], axis=0)
    down_ros = A_CV_TO_ROS @ (down_cv / np.linalg.norm(down_cv))
    return float(np.degrees(np.arccos(np.clip(-down_ros[2], -1, 1))))
