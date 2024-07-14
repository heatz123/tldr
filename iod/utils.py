import copy
import pathlib
import time
import os
import cv2
from PIL import Image
import io
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import torch
import platform

import dowel_wrapper

if "macOS" in platform.platform():
    os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/bin/ffmpeg"
from moviepy import editor as mpy
from garage.misc.tensor_utils import discount_cumsum
from matplotlib import figure
from matplotlib.patches import Ellipse
from sklearn import decomposition


def to_np_object_arr(x):
    arr = np.empty(len(x), dtype=object)
    for i, t in enumerate(x):
        arr[i] = t
    return arr


def get_torch_concat_obs(obs, option, dim=1):
    concat_obs = torch.cat([obs] + [option], dim=dim)
    return concat_obs


def get_np_concat_obs(obs, option):
    concat_obs = np.concatenate([obs] + [option])
    return concat_obs


def get_normalizer_preset(normalizer_type):
    # Precomputed mean and std of the state dimensions from 10000 length-50 random rollouts (without early termination)
    if normalizer_type == "off":
        normalizer_mean = np.array([0.0])
        normalizer_std = np.array([1.0])
    elif normalizer_type == "half_cheetah_preset":
        normalizer_mean = np.array(
            [
                -0.07861924,
                -0.08627162,
                0.08968642,
                0.00960849,
                0.02950368,
                -0.00948337,
                0.01661406,
                -0.05476654,
                -0.04932635,
                -0.08061652,
                -0.05205841,
                0.04500197,
                0.02638421,
                -0.04570961,
                0.03183838,
                0.01736591,
                0.0091929,
                -0.0115027,
            ]
        )
        normalizer_std = np.array(
            [
                0.4039283,
                0.07610687,
                0.23817,
                0.2515473,
                0.2698137,
                0.26374814,
                0.32229397,
                0.2896734,
                0.2774097,
                0.73060024,
                0.77360505,
                1.5871304,
                5.5405455,
                6.7097645,
                6.8253727,
                6.3142195,
                6.417641,
                5.9759197,
            ]
        )
    # elif normalizer_type == "ant_preset":
    elif "ant" in normalizer_type:
        normalizer_mean = np.array(
            [
                0.00486117,
                0.011312,
                0.7022248,
                0.8454677,
                -0.00102548,
                -0.00300276,
                0.00311523,
                -0.00139029,
                0.8607109,
                -0.00185301,
                -0.8556998,
                0.00343217,
                -0.8585605,
                -0.00109082,
                0.8558013,
                0.00278213,
                0.00618173,
                -0.02584622,
                -0.00599026,
                -0.00379596,
                0.00526138,
                -0.0059213,
                0.27686235,
                0.00512205,
                -0.27617684,
                -0.0033233,
                -0.2766923,
                0.00268359,
                0.27756855,
            ]
        )
        normalizer_std = np.array(
            [
                0.62473416,
                0.61958003,
                0.1717569,
                0.28629342,
                0.20020866,
                0.20572574,
                0.34922406,
                0.40098143,
                0.3114514,
                0.4024826,
                0.31057045,
                0.40343934,
                0.3110796,
                0.40245822,
                0.31100526,
                0.81786263,
                0.8166509,
                0.9870919,
                1.7525449,
                1.7468817,
                1.8596431,
                4.502961,
                4.4070187,
                4.522444,
                4.3518476,
                4.5105968,
                4.3704205,
                4.5175962,
                4.3704395,
            ]
        )
    elif (
        "quadruped_state_run_forward" in normalizer_type
        or "quadruped_state_step" in normalizer_type
    ):
        print("Using quadruped_state normalizer")
        normalizer_mean = np.array(
            [
                1.8927836e-03,
                2.4041167e-04,
                5.1787740e-01,
                -5.9818529e-04,
                -4.3730151e-02,
                3.9473318e-05,
                4.3242671e-02,
                -9.2326244e-04,
                -4.3722145e-02,
                -2.7089342e-04,
                4.3554552e-02,
                1.0351476e-03,
                -4.2407751e-02,
                5.7628180e-04,
                4.1396014e-02,
                2.9282683e-05,
                -4.0890679e-02,
                -1.9898126e-03,
                4.2425741e-02,
                -9.5242191e-05,
                -1.4839959e-02,
                2.9796332e-03,
                1.5355589e-02,
                -1.5644074e-03,
                -1.5756292e-02,
                4.2371741e-03,
                1.6236050e-02,
                -6.2402285e-04,
                -1.5238622e-02,
                3.7957507e-03,
                1.3513504e-02,
                4.1693915e-04,
                -1.4897966e-02,
                2.9975194e-03,
                1.4967091e-02,
                -7.6120475e-04,
                4.7387175e-02,
                -5.7184021e-04,
                -5.9542368e-04,
                4.6838216e-02,
                -3.3304640e-04,
                6.7474839e-04,
                4.8830710e-02,
                -7.9038856e-04,
                1.0581809e-04,
                4.9288359e-02,
                1.1667573e-03,
                1.6429248e-04,
                -6.0747399e-05,
                -7.8207068e-03,
                9.8630720e-01,
                -5.4306365e-03,
                2.5889047e-03,
                9.1699314e00,
                3.9346554e-04,
                -4.6278430e-05,
                1.4264234e-03,
                1.3697726e-01,
                -5.0663585e-03,
                -2.9545226e00,
                1.3929485e-01,
                1.3772677e-02,
                -2.9471116e00,
                1.4515232e-01,
                -7.2479318e-03,
                -2.9518723e00,
                1.3382623e-01,
                1.4965420e-03,
                -2.9518878e00,
                2.5628498e-03,
                -2.9405120e-01,
                2.2139077e-03,
                4.7230860e-03,
                -3.0089971e-01,
                -2.2757549e-03,
                -7.4373991e-03,
                -3.0217928e-01,
                -1.7959431e-03,
                8.5901184e-04,
                -2.9730809e-01,
                -7.4816146e-04,
            ]
        )
        normalizer_std = np.array(
            [
                0.20309114,
                0.20344722,
                0.07145283,
                0.23439685,
                0.20748195,
                0.19914377,
                0.23145178,
                0.23392929,
                0.2072366,
                0.20000164,
                0.23091461,
                0.23394057,
                0.20745817,
                0.1989382,
                0.23141295,
                0.23439825,
                0.20760228,
                0.19938056,
                0.23168479,
                3.0734482,
                2.2219946,
                2.158713,
                2.5998635,
                3.0752475,
                2.220529,
                2.1578412,
                2.600774,
                3.0758638,
                2.2195349,
                2.158783,
                2.5993125,
                3.0711286,
                2.2188709,
                2.1605713,
                2.601727,
                0.2575298,
                0.2706898,
                0.20579752,
                0.25723672,
                0.27015796,
                0.20617835,
                0.25706577,
                0.27049136,
                0.20558944,
                0.2580465,
                0.27033204,
                0.20577712,
                0.3958721,
                0.39333043,
                0.5491541,
                0.03021284,
                7.664319,
                7.632758,
                9.197585,
                1.1313747,
                1.1281061,
                1.2753158,
                5.1873183,
                5.0729136,
                4.4584656,
                5.18317,
                5.068774,
                4.456239,
                5.186024,
                5.06933,
                4.455007,
                5.187288,
                5.0716443,
                4.454951,
                2.2821507,
                2.5068493,
                1.258166,
                2.2772183,
                2.5008159,
                1.2556746,
                2.278745,
                2.5015957,
                1.2549113,
                2.2799993,
                2.5030627,
                1.2545576,
            ]
        )

    elif "humanoid_state" in normalizer_type:
        print("Using humanoid_state normalizer")
        use_pure_state = True
        if not use_pure_state:
            normalizer_mean = np.array(
                [
                    0,
                    0,
                    0,
                    -4.55341563e-02,
                    -1.21985875e-01,
                    -2.31404286e-02,
                    -1.20687634e-01,
                    -1.30207315e-01,
                    -3.64146411e-01,
                    -1.46858561e00,
                    -6.74302131e-02,
                    4.36150245e-02,
                    -1.20611079e-01,
                    -4.52108048e-02,
                    -3.83175284e-01,
                    -1.44170153e00,
                    -6.32304251e-02,
                    -2.05550808e-02,
                    -6.26289099e-02,
                    -2.40767345e-01,
                    -1.81927338e-01,
                    1.60777971e-01,
                    2.74439275e-01,
                    -2.00593978e-01,
                    2.65849918e-01,
                    1.51950985e-01,
                    2.44531021e-01,
                    2.22709104e-02,
                    -1.37690976e-02,
                    1.06678054e-01,
                    -7.69879341e-01,
                    1.63875684e-01,
                    -2.26381272e-01,
                    1.27363550e-02,
                    -3.04368120e-02,
                    -1.25551596e-01,
                    -7.65002131e-01,
                    1.27381712e-01,
                    1.91182524e-01,
                    5.24487421e-02,
                    4.80613783e-02,
                    -3.70609947e-02,
                    -2.62228757e-01,
                    -8.49810429e-04,
                    -1.82698946e-02,
                    -2.61723369e-01,
                    -1.09593824e-01,
                    -1.31301984e-01,
                    2.32464354e-02,
                    -9.93370730e-03,
                    1.91359863e-01,
                    7.04954797e-03,
                    -3.52000892e-02,
                    6.69755274e-03,
                    -5.48324874e-03,
                    1.20332547e-01,
                    2.05669791e-01,
                    7.30814114e-02,
                    -5.05045764e-02,
                    2.63460632e-02,
                    6.93454314e-03,
                    1.13557905e-01,
                    1.82707250e-01,
                    -3.43055949e-02,
                    2.81349686e-03,
                    4.94275019e-02,
                    -1.08685821e-01,
                    1.30921165e-02,
                    -6.21576048e-02,
                    -1.28641918e-01,
                ]
            )
            normalizer_std = np.array(
                [
                    1,
                    1,
                    1,
                    0.39989153,
                    0.45102388,
                    0.36428273,
                    0.19734234,
                    0.4788912,
                    0.61039275,
                    1.0132267,
                    0.6997092,
                    0.732887,
                    0.19663857,
                    0.45847908,
                    0.6240397,
                    1.01576,
                    0.70137686,
                    0.73841375,
                    0.89309657,
                    0.7507263,
                    0.96650773,
                    0.8984303,
                    0.7621492,
                    0.9669089,
                    0.32441196,
                    0.19762062,
                    0.21468523,
                    0.23483388,
                    0.31706256,
                    0.28273124,
                    0.2728141,
                    0.19288784,
                    0.20886853,
                    0.23346993,
                    0.30975088,
                    0.2807222,
                    0.26785284,
                    0.8088912,
                    0.47262207,
                    0.25818083,
                    0.29405755,
                    0.2947045,
                    0.89054364,
                    0.42628494,
                    0.43088505,
                    0.9609553,
                    1.8837638,
                    2.4032736,
                    3.974841,
                    4.02501,
                    4.2648506,
                    3.3945034,
                    3.229742,
                    3.9481041,
                    6.970455,
                    12.228988,
                    18.13325,
                    19.161802,
                    3.2377825,
                    3.9298086,
                    7.084742,
                    12.632224,
                    18.42646,
                    19.583532,
                    8.019335,
                    7.8995824,
                    14.993097,
                    8.520627,
                    8.296156,
                    15.488239,
                ]
            )

        else:
            normalizer_mean = np.array(
                [
                    0.03112607,
                    -0.07387013,
                    0.36554655,
                    0.4294679,
                    0.37751335,
                    0.25040394,
                    0.3111829,
                    -0.04045559,
                    -0.16843604,
                    -0.02147088,
                    -0.11596029,
                    -0.12373827,
                    -0.39522046,
                    -1.4752568,
                    -0.06423807,
                    0.04765055,
                    -0.11975417,
                    -0.03923087,
                    -0.41953984,
                    -1.4343326,
                    -0.05967266,
                    -0.01487488,
                    -0.00982091,
                    -0.24723615,
                    -0.19611897,
                    0.16084036,
                    0.26873136,
                    -0.2292702,
                    0.00434303,
                    -0.03359267,
                    -0.53370136,
                    -0.14816526,
                    -0.21436125,
                    0.03258101,
                    -0.02710315,
                    0.26091567,
                    -0.00445439,
                    -0.02341699,
                    0.02012537,
                    0.08066219,
                    0.08103184,
                    0.18835181,
                    0.0848648,
                    -0.03875607,
                    0.07146649,
                    0.1110159,
                    0.09872998,
                    0.17893036,
                    -0.03296712,
                    0.05619631,
                    0.03727197,
                    -0.05170243,
                    -0.00159632,
                    -0.03071329,
                    -0.07570447,
                ],
            )
            normalizer_std = np.array(
                [
                    0.25902084,
                    0.25744286,
                    0.42351952,
                    0.3099751,
                    0.32775223,
                    0.40969625,
                    0.37692466,
                    0.40411863,
                    0.47783136,
                    0.3660218,
                    0.19747983,
                    0.47304037,
                    0.6295187,
                    1.0103253,
                    0.6978385,
                    0.7295383,
                    0.19671917,
                    0.4449006,
                    0.64526975,
                    1.0172966,
                    0.7018878,
                    0.7379168,
                    0.88701844,
                    0.76501554,
                    0.962493,
                    0.8939886,
                    0.7726506,
                    0.96609426,
                    0.46957847,
                    0.46547437,
                    1.2297927,
                    1.9973532,
                    2.5657754,
                    4.104149,
                    4.1049695,
                    4.3795185,
                    3.5008314,
                    3.2681093,
                    4.005164,
                    6.9880257,
                    12.220136,
                    18.060743,
                    19.021585,
                    3.2737067,
                    3.9692059,
                    7.157682,
                    12.822448,
                    18.512821,
                    19.625578,
                    8.122318,
                    7.919952,
                    14.979987,
                    8.840243,
                    8.54118,
                    15.772103,
                ]
            )
    elif "quadruped_state_escape" in normalizer_type:
        print("Using quadruped_state_escape normalizer")
        normalizer_mean = np.array(
            [
                -8.20556656e-03,
                2.40530982e-03,
                3.79398495e-01,
                1.55653211e-03,
                4.20332253e-02,
                -1.54986233e-02,
                -2.68592443e-02,
                -9.13381285e-04,
                4.27528210e-02,
                -1.65221561e-02,
                -2.65552104e-02,
                -1.63943158e-03,
                4.28224839e-02,
                -1.62665062e-02,
                -2.68660728e-02,
                -3.31533141e-04,
                4.22220677e-02,
                -1.53937694e-02,
                -2.71427538e-02,
                -3.92379035e-04,
                8.15717969e-03,
                -9.30575002e-03,
                -1.14900367e-02,
                -1.37746357e-03,
                8.34245794e-03,
                -8.44223890e-03,
                -1.20588979e-02,
                -7.98371213e-04,
                1.01060923e-02,
                -8.32669344e-03,
                -1.21675627e-02,
                -1.35593221e-03,
                8.56021605e-03,
                -7.11246254e-03,
                -1.42010739e-02,
                1.59397349e-03,
                4.90553491e-02,
                2.49159028e-04,
                -9.91887529e-04,
                4.89886850e-02,
                8.88779643e-04,
                -1.52114686e-03,
                4.94292341e-02,
                7.73700827e-04,
                -5.40030363e-04,
                4.89149578e-02,
                -2.94554666e-05,
                -4.21465025e-04,
                -6.66248263e-04,
                1.01498455e-01,
                -4.73929137e-01,
                1.01841344e-02,
                6.76223543e-03,
                -4.75091171e00,
                1.07999018e-03,
                4.81115608e-03,
                3.79599514e-03,
                -2.62875885e-01,
                -2.88672769e-03,
                -9.49159265e-01,
                -2.75353402e-01,
                -1.65101839e-03,
                -9.43003416e-01,
                -2.61171997e-01,
                1.29202311e-03,
                -9.53744888e-01,
                -2.60383785e-01,
                -3.11638624e-03,
                -9.45809841e-01,
                -2.52289488e-03,
                -4.75242808e-02,
                -2.02223819e-04,
                -3.73172457e-04,
                -4.31095138e-02,
                -6.82684360e-04,
                -9.41891049e-04,
                -5.25226146e-02,
                -1.05728046e-03,
                2.17561537e-04,
                -4.89490628e-02,
                4.05956380e-04,
                8.07770807e-03,
                -1.74607872e-03,
                3.46445367e-02,
                8.93775165e-01,
                8.94708574e-01,
                8.94985318e-01,
                8.94552946e-01,
                8.93441916e-01,
                9.47988868e-01,
                9.47988868e-01,
                9.50376630e-01,
                9.49705005e-01,
                9.47826385e-01,
                9.52005386e-01,
                9.50575352e-01,
                9.48971868e-01,
                9.50554371e-01,
                9.51943159e-01,
                8.40262413e-01,
                9.13053691e-01,
                9.12544191e-01,
                9.13005233e-01,
                8.40034425e-01,
            ]
        )
        normalizer_std = np.array(
            [
                0.43406355,
                0.42486626,
                0.19773862,
                0.24444185,
                0.2180477,
                0.20257592,
                0.23707913,
                0.24486504,
                0.218547,
                0.20364511,
                0.23716547,
                0.24514833,
                0.21871892,
                0.20326902,
                0.23724069,
                0.2444065,
                0.21941522,
                0.20442635,
                0.23814356,
                3.4264126,
                2.4234133,
                2.1889617,
                2.786824,
                3.4344673,
                2.4293175,
                2.1867857,
                2.7911437,
                3.4286346,
                2.4252617,
                2.1909359,
                2.7836056,
                3.4257348,
                2.426208,
                2.191693,
                2.7886086,
                0.25719944,
                0.26988736,
                0.20546702,
                0.25729537,
                0.27030784,
                0.20577201,
                0.25789368,
                0.26988885,
                0.20574692,
                0.2574493,
                0.2704906,
                0.20634118,
                0.36122563,
                0.36370477,
                0.50909746,
                0.8022595,
                11.45137,
                11.502441,
                17.888788,
                1.241205,
                1.2287669,
                1.2251754,
                4.4116974,
                4.431034,
                4.217395,
                4.406584,
                4.428491,
                4.2191677,
                4.4110684,
                4.4281287,
                4.217087,
                4.409103,
                4.4291854,
                4.2152796,
                1.0699797,
                1.3153256,
                0.77918094,
                1.0672296,
                1.3099797,
                0.7748754,
                1.071123,
                1.3154134,
                0.77744865,
                1.0658289,
                1.3113236,
                0.7769675,
                0.46419257,
                0.46181515,
                0.34931657,
                0.21632609,
                0.21640874,
                0.21652903,
                0.21634738,
                0.21626806,
                0.15165178,
                0.15165178,
                0.15005074,
                0.15037437,
                0.15171424,
                0.12607104,
                0.12637472,
                0.1286381,
                0.12635024,
                0.1262051,
                0.19758113,
                0.16053525,
                0.16872676,
                0.16051482,
                0.19731739,
            ]
        )
    else:
        raise NotImplementedError

    return normalizer_mean, normalizer_std


def get_2d_colors(points, min_point, max_point):
    points = np.array(points)
    min_point = np.array(min_point)
    max_point = np.array(max_point)

    colors = (points - min_point) / (max_point - min_point)
    colors = np.hstack(
        (
            colors,
            (2 - np.sum(colors, axis=1, keepdims=True)) / 2,
        )
    )
    colors = np.clip(colors, 0, 1)
    colors = np.c_[colors, np.full(len(colors), 0.8)]

    return colors


def get_option_colors(options, color_range=4):
    num_options = options.shape[0]
    dim_option = options.shape[1]

    if dim_option <= 2:
        # Use a predefined option color scheme
        if dim_option == 1:
            options_2d = []
            d = 2.0
            for i in range(len(options)):
                option = options[i][0]
                if option < 0:
                    abs_value = -option
                    options_2d.append((d - abs_value * d, d))
                else:
                    abs_value = option
                    options_2d.append((d, d - abs_value * d))
            options = np.array(options_2d)
        option_colors = get_2d_colors(
            options, (-color_range, -color_range), (color_range, color_range)
        )
    else:
        if dim_option > 3 and num_options >= 3:
            pca = decomposition.PCA(n_components=3)
            # Add random noises to break symmetry.
            pca_options = np.vstack((options, np.random.randn(dim_option, dim_option)))
            pca.fit(pca_options)
            option_colors = np.array(pca.transform(options))
        elif dim_option > 3 and num_options < 3:
            option_colors = options[:, :3]
        elif dim_option == 3:
            option_colors = options

        max_colors = np.array([color_range] * 3)
        min_colors = np.array([-color_range] * 3)
        if all((max_colors - min_colors) > 0):
            option_colors = (option_colors - min_colors) / (max_colors - min_colors)
        option_colors = np.clip(option_colors, 0, 1)

        option_colors = np.c_[option_colors, np.full(len(option_colors), 0.8)]

    return option_colors


def draw_2d_gaussians(
    means,
    stddevs,
    colors,
    ax,
    fill=False,
    alpha=0.8,
    use_adaptive_axis=False,
    draw_unit_gaussian=True,
    plot_axis=None,
):
    means = np.clip(means, -1000, 1000)
    stddevs = np.clip(stddevs, -1000, 1000)

    square_axis_limit = 2.0
    if draw_unit_gaussian:
        ellipse = Ellipse(
            xy=(0, 0),
            width=2,
            height=2,
            edgecolor="r",
            lw=1,
            facecolor="none",
            alpha=0.5,
        )
        ax.add_patch(ellipse)
    for mean, stddev, color in zip(means, stddevs, colors):
        if len(mean) == 1:
            mean = np.concatenate([mean, [0.0]])
            stddev = np.concatenate([stddev, [0.1]])
        # even if # mean dim > 2, it only chooses first 2 dimensions
        ellipse = Ellipse(
            xy=mean,
            width=stddev[0] * 2,
            height=stddev[1] * 2,
            edgecolor=color,
            lw=1,
            facecolor="none" if not fill else color,
            alpha=alpha,
        )
        ax.add_patch(ellipse)
        square_axis_limit = max(
            square_axis_limit,
            np.abs(mean[0] + stddev[0]),
            np.abs(mean[0] - stddev[0]),
            np.abs(mean[1] + stddev[1]),
            np.abs(mean[1] - stddev[1]),
        )
    square_axis_limit = square_axis_limit * 1.2
    ax.axis("scaled")
    if plot_axis is None:
        if use_adaptive_axis:
            ax.set_xlim(-square_axis_limit, square_axis_limit)
            ax.set_ylim(-square_axis_limit, square_axis_limit)
        else:
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
    else:
        ax.axis(plot_axis)


def prepare_video(v, n_cols=None):
    orig_ndim = v.ndim
    if orig_ndim == 4:
        v = v[None,]

    _, t, c, h, w = v.shape

    if v.dtype == np.uint8:
        v = np.float32(v) / 255.0

    def is_power2(num):
        return num != 0 and ((num & (num - 1)) == 0)

    if n_cols is None:
        if v.shape[0] <= 3:
            n_cols = v.shape[0]
        elif v.shape[0] <= 9:
            n_cols = 3
        else:
            n_cols = 6
    if v.shape[0] % n_cols != 0:
        len_addition = n_cols - v.shape[0] % n_cols
        v = np.concatenate((v, np.zeros(shape=(len_addition, t, c, h, w))), axis=0)
    n_rows = v.shape[0] // n_cols

    v = np.reshape(v, newshape=(n_rows, n_cols, t, c, h, w))
    v = np.transpose(v, axes=(2, 0, 4, 1, 5, 3))
    v = np.reshape(v, newshape=(t, n_rows * h, n_cols * w, c))

    return v


def save_video(runner, label, tensor, fps=15, n_cols=None):
    def _to_uint8(t):
        # If user passes in uint8, then we don't need to rescale by 255
        if t.dtype != np.uint8:
            t = (t * 255.0).astype(np.uint8)
        return t

    if tensor.dtype in [object]:
        tensor = [_to_uint8(prepare_video(t, n_cols)) for t in tensor]
    else:
        assert tensor.shape[2] == 3, tensor.shape
        tensor = prepare_video(tensor, n_cols)
        tensor = _to_uint8(tensor)

    # Encode sequence of images into gif string
    clip = mpy.ImageSequenceClip(list(tensor), fps=fps)

    plot_path = (
        pathlib.Path(runner._snapshotter.snapshot_dir)
        / "plots"
        # / f'{label}_{runner.step_itr}.gif')
        / f"{label}_{runner.step_itr}.mp4"
    )
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    clip.write_videofile(str(plot_path), audio=False, verbose=False, logger=None)

    if "WANDB_API_KEY" in os.environ:
        import wandb

        wandb.log({label: wandb.Video(str(plot_path))}, step=runner.step_itr)


def record_video(
    runner, label, trajectories, n_cols=None, skip_frames=1, phi=None, goal_images=None
):
    assert "agent_infos" in trajectories[0], trajectories[0].keys()
    assert "env_infos" in trajectories[0], trajectories[0].keys()
    # desired format for renders is [frames 3 height width]
    renders = []
    for trajectory in trajectories:
        render = trajectory["env_infos"]["render"]
        if render.ndim >= 5:
            render = render.reshape(-1, *render.shape[-3:])
        elif render.ndim == 1:  # which case?
            render = np.concatenate(render, axis=0)
        renders.append(render)
    max_length = max([len(render) for render in renders])

    for i, render in enumerate(renders):
        assert render.ndim == 4  # [frames 3 height width]
        renders[i] = np.concatenate(
            [
                render,
                np.zeros(
                    (max_length - render.shape[0], *render.shape[1:]),
                    dtype=render.dtype,
                ),
            ],
            axis=0,
        )
        renders_marked = renders[i].copy()
        renders_marked[:, :, :10, -10:] = np.array([255, 0, 0])[
            :, None, None
        ]  # fill red
        cur_exploration = trajectories[i]["agent_infos"]["cur_exploration"]
        assert cur_exploration.shape == (len(renders[i]),), (
            cur_exploration.shape,
            len(renders[i]),
            len(renders),
            # this happens when timesteps are not fixed
        )
        renders[i] = np.where(
            cur_exploration[:, None, None, None], renders_marked, renders[i]
        )

        renders[i] = renders[i][::skip_frames]
    renders = np.array(renders, dtype=np.uint8)
    assert renders.ndim == 5

    if goal_images is not None:
        if goal_images.ndim != 5:
            assert goal_images.ndim == 4 and goal_images.shape[0] == len(
                renders
            ), goal_images.shape
            # fill green
            goal_images[:, :, :10, -10:] = np.array([0, 255, 0])[None, :, None, None]

            goal_images = np.repeat(
                goal_images[:, None, :, :, :], len(renders[0]), axis=1
            )
        N, L, c, h, w = renders.shape
        if goal_images.shape[-1] != renders.shape[-1]:

            # resize goal_images to match renders
            # Function to resize a single frame

            # Resize each frame in the video
            goal_images = goal_images.transpose(0, 1, 3, 4, 2)  # N L h w c
            goal_images = np.array(
                [
                    [
                        resize_with_padding(goal_images[n, l], desired_size=h)
                        for l in range(L)
                    ]
                    for n in range(N)
                ],
                dtype=np.uint8,
            )
            goal_images = goal_images.transpose(0, 1, 4, 2, 3)

            assert renders[0].dtype == goal_images[0].dtype, (
                renders[0].dtype,
                goal_images[0].dtype,
            )
            assert renders.shape[1] == goal_images.shape[1], (
                renders[0].shape,
                renders[1].shape,
                goal_images[0].shape,
                goal_images[1].shape,
            )
        # interleave goal_images and renders
        renders = (
            np.stack([renders, goal_images], axis=0)
            .swapaxes(0, 1)
            .reshape(-1, L, c, h, w)
        )  # N * 2, C, H, W

    if phi is not None:
        assert len(trajectories) == 1
        assert len(phi) == 1, f"phi shape: {phi.shape}"

        phi = phi[0]

        images = []
        for i in range(len(phi)):
            # plot phi as 2d gaussians in 2d array dynamically
            fig, ax = plt.subplots()

            # Choose your colormap
            cmap = cm.get_cmap("tab10")

            # Generate a sequence of numbers between 0 and 1 to map to the colormap
            indices = np.linspace(0, 1, len(phi))

            # Map the sequence of numbers to colors in the colormap
            colors = cmap(indices)[:, :3]

            draw_2d_gaussians(
                phi[: i + 1, :],
                [[0.03, 0.03]] * len(phi[: i + 1, :]),
                colors[: i + 1],
                ax,
                fill=True,
                use_adaptive_axis=True,
            )
            plt.close(fig)  # Close the figure to prevent it from displaying

            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)

            image = Image.open(buf)
            image_array = np.array(image)[:, :, :3]
            images.append(image_array)
            buf.close()

        images = np.stack(images, axis=0)

        renders = renders[0].transpose(0, 2, 3, 1)
        assert renders.shape[-1] == 3, renders.shape
        new_height, new_width = images.shape[-3], images.shape[-2]
        resized_video = np.zeros_like(images)
        # Resize each frame
        for i in range(images.shape[0]):
            resized_video[i, :, :, :] = cv2.resize(renders[i], (new_width, new_height))
        renders = np.concatenate(
            [resized_video, images], axis=2
        )  # [frames height width 3]
        renders = renders.transpose(0, 3, 1, 2)[None]

        save_video(runner, label, renders, n_cols=n_cols)
    else:
        save_video(runner, label, renders, n_cols=n_cols)


def resize_with_padding(image, desired_size=256):
    old_size = image.shape[:2]  # old_size is in (height, width) format

    # Calculate the ratio to resize the image
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # Resize the image
    resized_image = cv2.resize(image, (new_size[1], new_size[0]))

    # Calculate padding
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    # Add padding
    color = [255, 255, 255]  # Black padding
    new_image = cv2.copyMakeBorder(
        resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )

    return new_image


class FigManager:
    def __init__(self, runner, label, extensions=None, subplot_spec=None):
        self.runner = runner
        self.label = label
        self.fig = figure.Figure()
        if subplot_spec is not None:
            self.ax = self.fig.subplots(*subplot_spec).flatten()
        else:
            self.ax = self.fig.add_subplot()

        if extensions is None:
            self.extensions = ["png"]
        else:
            self.extensions = extensions

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        plot_paths = [
            (
                pathlib.Path(self.runner._snapshotter.snapshot_dir)
                / "plots"
                / f"{self.label}_{self.runner.step_itr}.{extension}"
            )
            for extension in self.extensions
        ]
        plot_paths[0].parent.mkdir(parents=True, exist_ok=True)
        for plot_path in plot_paths:
            self.fig.savefig(plot_path, dpi=300)
        dowel_wrapper.get_tabular("plot").record(self.label, self.fig)


class MeasureAndAccTime:
    def __init__(self, target):
        assert isinstance(target, list)
        assert len(target) == 1
        self._target = target

    def __enter__(self):
        self._time_enter = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._target[0] += time.time() - self._time_enter


class Timer:
    def __init__(self):
        self.t = time.time()

    def __call__(self, msg="", *args, **kwargs):
        print(f"{msg}: {time.time() - self.t:.20f}")
        self.t = time.time()


def valuewise_sequencify_dicts(dicts):
    result = dict((k, []) for k in dicts[0].keys())
    for d in dicts:
        for k, v in d.items():
            result[k].append(v)
    return result


def zip_dict(d):
    keys = list(d.keys())
    values = [d[k] for k in keys]
    for z in zip(*values):
        yield dict((k, v) for k, v in zip(keys, z))


def split_paths(paths, chunking_points):
    assert 0 in chunking_points
    assert len(chunking_points) >= 2
    if len(chunking_points) == 2:
        return

    orig_paths = copy.copy(paths)
    paths.clear()
    for path in orig_paths:
        ei = path
        for s, e in zip(chunking_points[:-1], chunking_points[1:]):
            assert (
                len(
                    set(
                        len(v)
                        for k, v in path.items()
                        if k not in ["env_infos", "agent_infos"]
                    )
                )
                == 1
            )
            new_path = {
                k: v[s:e]
                for k, v in path.items()
                if k not in ["env_infos", "agent_infos"]
            }
            new_path["dones"][-1] = True

            assert len(set(len(v) for k, v in path["env_infos"].items())) == 1
            new_path["env_infos"] = {k: v[s:e] for k, v in path["env_infos"].items()}

            assert len(set(len(v) for k, v in path["agent_infos"].items())) == 1
            new_path["agent_infos"] = {
                k: v[s:e] for k, v in path["agent_infos"].items()
            }

            paths.append(new_path)


def compute_traj_batch_performance(batch, discount):
    returns = []
    undiscounted_returns = []
    for trajectory in batch.split():
        returns.append(discount_cumsum(trajectory.rewards, discount))
        undiscounted_returns.append(sum(trajectory.rewards))

    return dict(
        undiscounted_returns=undiscounted_returns,
        discounted_returns=[rtn[0] for rtn in returns],
    )
