from model.defer import DEFER5to30Model, DEFER5to90Model


class ESDFM5to30Model(DEFER5to30Model):
    """
    ES-DFM-5-30:
    - 5min 为 observation window
    - 30min 为 attribution window
    - 读入 5min-30min conditional emission 数据流

    与 DEFER 的 tower、fake-negative loss 和时长预估公式一致；
    差异在于 conditional emission 数据流中，label < 5min 的样本不会再进入
    二阶段窗口，因此这些样本只在 5min 到达时一次性完成所有 tower 的更新。
    """

    name = "ES-DFM-5-30"


class ESDFM5to90Model(DEFER5to90Model):
    """
    ES-DFM-5-90:
    - 5min 为 observation window
    - 90min 为 attribution window
    - 读入 5min-90min conditional emission 数据流

    与 DEFER 的 tower、fake-negative loss 和时长预估公式一致；
    差异在于 conditional emission 数据流中，label < 5min 的样本不会再进入
    二阶段窗口，因此这些样本只在 5min 到达时一次性完成所有 tower 的更新。
    """

    name = "ES-DFM-5-90"
