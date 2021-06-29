import json
import codecs
import os
import time
import sys

import tensorflow as tf
import numpy as np

from fusion import Fusion

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"


def fusion_checkpoint(checkpoint_paths: list, save_model_dir: str):
    new_var_list = []
    checkpoint_path = checkpoint_paths[0]
    os.system("rm -rf "+save_model_dir)
    os.system("mkdir "+save_model_dir)
    # print("开始融合checkpoint")
    # Num = 0
    for var_name, _ in tf.contrib.framework.list_variables(checkpoint_path):  # 得到checkpoint文件中所有的参数（名字，形状）元组
        # print(Num/112)
        # Num+=1
        temp_list = []
        for checkpoint_path in checkpoint_paths:
            var = tf.contrib.framework.load_variable(checkpoint_path, var_name)  # 得到上述参数的值
            temp_list.append(var)
        new_var = np.array(temp_list).mean(axis=0)
        renamed_var = tf.Variable(new_var, name=var_name)  # 使用加入前缀的新名称重新构造了参数
        new_var_list.append(renamed_var)

    with tf.Session() as sess:
        saver = tf.train.Saver(var_list=new_var_list)  # 构造一个保存器
        sess.run(tf.global_variables_initializer())  # 初始化一下参数（这一步必做）
        model_name = 'muti_checkpoint'  # 构造一个保存的模型名称
        checkpoint_path = os.path.join(save_model_dir, model_name)  # 构造一下保存路径
        saver.save(sess, checkpoint_path)  # 直接进行保存
    # print("checkpoint融合完成，保存在tangle下")


def main(model_dir: str, n: int, new_model_dir: str):
    checkpoint_list = ['model_epoch_112_gs_101472', 'model_epoch_111_gs_100566', 'model_epoch_110_gs_99660', 'model_epoch_109_gs_98754', 'model_epoch_108_gs_97848', 'model_epoch_107_gs_96942', 'model_epoch_106_gs_96036', 'model_epoch_105_gs_95130', 'model_epoch_104_gs_94224', 'model_epoch_103_gs_93318', 'model_epoch_102_gs_92412', 'model_epoch_101_gs_91506', 'model_epoch_100_gs_90600', 'model_epoch_99_gs_89694', 'model_epoch_98_gs_88788', 'model_epoch_97_gs_87882', 'model_epoch_96_gs_86976', 'model_epoch_95_gs_86070', 'model_epoch_94_gs_85164', 'model_epoch_93_gs_84258', 'model_epoch_92_gs_83352', 'model_epoch_91_gs_82446', 'model_epoch_90_gs_81540', 'model_epoch_89_gs_80634', 'model_epoch_88_gs_79728', 'model_epoch_87_gs_78822', 'model_epoch_86_gs_77916', 'model_epoch_85_gs_77010', 'model_epoch_84_gs_76104', 'model_epoch_83_gs_75198', 'model_epoch_82_gs_74292', 'model_epoch_81_gs_73386', 'model_epoch_80_gs_72480', 'model_epoch_79_gs_71574', 'model_epoch_78_gs_70668', 'model_epoch_77_gs_69762', 'model_epoch_76_gs_68856', 'model_epoch_75_gs_67950', 'model_epoch_74_gs_67044', 'model_epoch_73_gs_66138', 'model_epoch_72_gs_65232', 'model_epoch_71_gs_64326', 'model_epoch_70_gs_63420', 'model_epoch_69_gs_62514', 'model_epoch_68_gs_61608', 'model_epoch_67_gs_60702', 'model_epoch_66_gs_59796', 'model_epoch_65_gs_58890', 'model_epoch_64_gs_57984', 'model_epoch_63_gs_57078', 'model_epoch_62_gs_56172', 'model_epoch_61_gs_55266', 'model_epoch_60_gs_54360', 'model_epoch_59_gs_53454', 'model_epoch_58_gs_52548', 'model_epoch_57_gs_51642', 'model_epoch_56_gs_50736', 'model_epoch_55_gs_49830', 'model_epoch_54_gs_48924', 'model_epoch_53_gs_48018', 'model_epoch_52_gs_47112', 'model_epoch_51_gs_46206', 'model_epoch_50_gs_45300', 'model_epoch_49_gs_44394', 'model_epoch_48_gs_43488', 'model_epoch_47_gs_42582', 'model_epoch_46_gs_41676', 'model_epoch_45_gs_40770', 'model_epoch_44_gs_39864', 'model_epoch_43_gs_38958', 'model_epoch_42_gs_38052', 'model_epoch_41_gs_37146', 'model_epoch_40_gs_36240', 'model_epoch_39_gs_35334', 'model_epoch_38_gs_34428', 'model_epoch_37_gs_33522', 'model_epoch_36_gs_32616', 'model_epoch_35_gs_31710', 'model_epoch_34_gs_30804', 'model_epoch_33_gs_29898', 'model_epoch_32_gs_28992', 'model_epoch_31_gs_28086', 'model_epoch_30_gs_27180', 'model_epoch_29_gs_26274', 'model_epoch_28_gs_25368', 'model_epoch_27_gs_24462', 'model_epoch_26_gs_23556', 'model_epoch_25_gs_22650', 'model_epoch_24_gs_21744', 'model_epoch_23_gs_20838', 'model_epoch_22_gs_19932', 'model_epoch_21_gs_19026', 'model_epoch_20_gs_18120', 'model_epoch_19_gs_17214', 'model_epoch_18_gs_16308', 'model_epoch_17_gs_15402', 'model_epoch_16_gs_14496', 'model_epoch_15_gs_13590', 'model_epoch_14_gs_12684', 'model_epoch_13_gs_11778', 'model_epoch_12_gs_10872', 'model_epoch_11_gs_9966', 'model_epoch_10_gs_9060', 'model_epoch_09_gs_8154', 'model_epoch_08_gs_7248', 'model_epoch_07_gs_6342', 'model_epoch_06_gs_5436', 'model_epoch_05_gs_4530', 'model_epoch_04_gs_3624', 'model_epoch_03_gs_2718', 'model_epoch_02_gs_1812', 'model_epoch_01_gs_906']


    checkpoint_list = ['model_epoch_112_gs_101472', 'model_epoch_111_gs_100566', 'model_epoch_110_gs_99660', 'model_epoch_109_gs_98754', 'model_epoch_108_gs_97848', 'model_epoch_107_gs_96942', 'model_epoch_106_gs_96036', 'model_epoch_105_gs_95130', 'model_epoch_104_gs_94224', 'model_epoch_103_gs_93318', 'model_epoch_102_gs_92412', 'model_epoch_101_gs_91506', 'model_epoch_100_gs_90600', 'model_epoch_99_gs_89694', 'model_epoch_98_gs_88788', 'model_epoch_97_gs_87882', 'model_epoch_96_gs_86976', 'model_epoch_95_gs_86070', 'model_epoch_94_gs_85164', 'model_epoch_93_gs_84258', 'model_epoch_92_gs_83352', 'model_epoch_91_gs_82446', 'model_epoch_90_gs_81540', 'model_epoch_89_gs_80634', 'model_epoch_88_gs_79728', 'model_epoch_87_gs_78822', 'model_epoch_86_gs_77916', 'model_epoch_85_gs_77010', 'model_epoch_84_gs_76104', 'model_epoch_83_gs_75198', 'model_epoch_82_gs_74292', 'model_epoch_81_gs_73386', 'model_epoch_80_gs_72480', 'model_epoch_79_gs_71574', 'model_epoch_78_gs_70668', 'model_epoch_77_gs_69762', 'model_epoch_76_gs_68856', 'model_epoch_75_gs_67950', 'model_epoch_74_gs_67044', 'model_epoch_73_gs_66138', 'model_epoch_72_gs_65232', 'model_epoch_71_gs_64326', 'model_epoch_70_gs_63420', 'model_epoch_69_gs_62514', 'model_epoch_68_gs_61608', 'model_epoch_67_gs_60702', 'model_epoch_66_gs_59796', 'model_epoch_65_gs_58890', 'model_epoch_64_gs_57984', 'model_epoch_63_gs_57078', 'model_epoch_62_gs_56172', 'model_epoch_61_gs_55266', 'model_epoch_60_gs_54360', 'model_epoch_59_gs_53454', 'model_epoch_58_gs_52548', 'model_epoch_57_gs_51642', 'model_epoch_56_gs_50736', 'model_epoch_55_gs_49830', 'model_epoch_54_gs_48924', 'model_epoch_53_gs_48018', 'model_epoch_52_gs_47112', 'model_epoch_51_gs_46206', 'model_epoch_50_gs_45300', 'model_epoch_49_gs_44394', 'model_epoch_48_gs_43488', 'model_epoch_47_gs_42582', 'model_epoch_46_gs_41676', 'model_epoch_45_gs_40770', 'model_epoch_44_gs_39864', 'model_epoch_43_gs_38958', 'model_epoch_42_gs_38052', 'model_epoch_41_gs_37146', 'model_epoch_40_gs_36240', 'model_epoch_39_gs_35334', 'model_epoch_38_gs_34428', 'model_epoch_37_gs_33522', 'model_epoch_36_gs_32616', 'model_epoch_35_gs_31710', 'model_epoch_34_gs_30804', 'model_epoch_33_gs_29898', 'model_epoch_32_gs_28992', 'model_epoch_31_gs_28086', 'model_epoch_30_gs_27180', 'model_epoch_29_gs_26274', 'model_epoch_28_gs_25368', 'model_epoch_27_gs_24462', 'model_epoch_26_gs_23556', 'model_epoch_25_gs_22650', 'model_epoch_24_gs_21744', 'model_epoch_23_gs_20838', 'model_epoch_22_gs_19932', 'model_epoch_21_gs_19026', 'model_epoch_20_gs_18120', 'model_epoch_19_gs_17214', 'model_epoch_18_gs_16308', 'model_epoch_17_gs_15402', 'model_epoch_16_gs_14496', 'model_epoch_15_gs_13590', 'model_epoch_14_gs_12684', 'model_epoch_13_gs_11778', 'model_epoch_12_gs_10872', 'model_epoch_11_gs_9966', 'model_epoch_10_gs_9060', 'model_epoch_09_gs_8154', 'model_epoch_08_gs_7248', 'model_epoch_07_gs_6342', 'model_epoch_06_gs_5436', 'model_epoch_05_gs_4530', 'model_epoch_04_gs_3624', 'model_epoch_03_gs_2718', 'model_epoch_02_gs_1812', 'model_epoch_01_gs_906']

    checkpoint_list = [model_dir + "/" + checkpoint for checkpoint in checkpoint_list]

    fusion_checkpoint(checkpoint_list[0:n], new_model_dir)


if __name__ == '__main__':
    model_dir = "tangle01"
    new_model_dir = "tangle"
    main(model_dir, 112, new_model_dir)

    # 计算得分
    f = Fusion([new_model_dir])
    f.eval(use_image=True)
