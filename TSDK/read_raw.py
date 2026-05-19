import os
# import opencv as cv2
import cv2
import numpy as np

# savepath = "C:/Users/poppip/Desktop/test/tempture/DJI_202207221723_011_/"
savepath = "/home/nyh/LY/dji_thermal_sdk_v1.7_OK/raw/"


def raw_to_tif(path, rows, cols, channels):
    print('to .tif start')
    files = os.listdir(path)
    for file in files:
        portion = os.path.splitext(file)
        if portion[1] == '.raw':
            realPath = path + file
            img = np.fromfile(realPath, dtype='uint16')
            img = img / 10  ##除10之后为温度值
            img = img.reshape(rows, cols, channels)
            fileName = portion[0] + '.tif'
            tif_fileName = os.path.join(path, fileName)
            cv2.imwrite(tif_fileName, img, (int(cv2.IMWRITE_TIFF_COMPRESSION), 1))
            os.remove(realPath)  ##delete .raw file，如有需要，可以不删除
        else:
            print(file + ' it is not .raw file')
    print('to .tif finsh')


raw_to_tif(savepath, 512, 640, 1)







































# # https://blog.csdn.net/qq_23865133/article/details/135813617
#
#
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import cm
#
# # 参数设置（需根据实际调整）
# width, height = 1280, 1024  # 红外图像分辨率（参考大疆H20T）
# dtype = np.float32        # 温度数据格式
# scale_factor = 0.01       # 若为uint16可能需要缩放因子
#
# # 读取RAW文件
# with open('measure.raw', 'rb') as f:
#     temp_data = np.fromfile(f, dtype=dtype).reshape((height, width))
#
# # 若为uint16转float温度值
# if dtype == np.uint16:
#     temp_data = temp_data * scale_factor
#
# # 显示温度矩阵
# plt.figure(figsize=(12, 6))
# plt.imshow(temp_data, cmap=cm.plasma, interpolation='nearest')
# plt.colorbar(label='Temperature (°C)')
# plt.title('DJI Thermal Image')
# plt.axis('off')
#
# # 标记温度极值
# max_temp = np.max(temp_data)
# min_temp = np.min(temp_data)
# plt.text(10, 30, f'Max: {max_temp:.1f}°C', color='white', fontsize=12)
# plt.text(10, 60, f'Min: {min_temp:.1f}°C', color='white', fontsize=12)
#
# plt.show()
#
