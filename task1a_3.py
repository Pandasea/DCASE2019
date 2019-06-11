import numpy as np
import csv
import pickle

mode = 'eval'  #两种模式'dev' or 'eval'，  'dev'输出开发集准确率，'eval'输出提交的结果

#四个输出文件，前三个为要提交的结果，最后一个是，development上各类场景以及平均的识别准确率
fusion_csvName  = './result/fusion_result.csv'
l_csvName  = './result/l_result.csv'
m_csvName  = './result/m_result.csv'
dev_result = './result/dev_result.csv'

#混淆矩阵
def matrix(dev_pred):
    label_path = 'testlabel_9_t.pkl'
    lf = open(label_path, 'rb')
    dev_label = pickle.load(lf)
    lf.close()
    dev_label = np.argmax(dev_label, 1)

    dev_matrix = [0 for _ in range(10)]
    count_matrix = [0 for _ in range(10)]
    counter = 0
    for i in range(2880):
        label = dev_label[i]
        pred = dev_pred[i]
        if label == pred:
            dev_matrix[label] = dev_matrix[label] + 1
            counter = counter + 1
        count_matrix[label] = count_matrix[label] + 1
    accu_list = []
    for i in range(10):
        accu = dev_matrix[i] / count_matrix[i]
        accu_list.append(accu)
    all_accu = counter/2880

    return accu_list,all_accu

#模型融合
def fusion(l_softmax, m_softmax, mode):
    l_pred = np.argmax(l_softmax, 1)
    m_pred = np.argmax(m_softmax, 1)
    fusion_pred = []
    if mode == 'dev':
        num = 2880
    else:
        num = 7200
    for i in range(num):
        if l_pred[i] == m_pred[i]:
            fusion_pred.append(l_pred[i])
        else:
            fusion_softmax = (l_softmax[i] + m_softmax[i]) / 2
            fusion_pred.append(np.argmax(fusion_softmax))

    return fusion_pred

#将结果写入csv中
def write_csv(csvName, pred_result):

    csvFile = open(csvName, 'w', newline='')  # 设置newline，否则两行之间会空一行
    writer = csv.writer(csvFile)
    scene_dic = {'0': 'airport', '1': 'shopping_mall', '2': 'metro_station', '3': 'street_pedestrian',
                 '4': 'public_square',
                 '5': 'street_traffic', '6': 'tram', '7': 'bus', '8': 'metro', '9': 'park'}
    for i in range(7200):
        result = pred_result[i]
        result = scene_dic[str(result)]
        result_writer = [str(i) + '.wav' + '    ' + result]
        writer.writerow(result_writer)
    csvFile.close()

#思凡的预测结果读入
m_dev_path = './softmax/train_2880_softmax.npy'
m_dev_softmax = np.load(m_dev_path)
m_eval_path = './softmax/dss_7200_softmax.npy'
m_eval_softmax = np.load(m_eval_path)
m_dev_pred  = np.argmax(m_dev_softmax, 1)
m_eval_pred = np.argmax(m_eval_softmax, 1)

#刘伟的预测结果读入
l_dev_softmax_path = 'dev_softmax.csv'
lf = open(l_dev_softmax_path,'r')
csv_reader = csv.reader(lf)
l_dev_softmax =[]
for row in csv_reader:
    l_dev_softmax.append(row)
lf.close()
l_dev_softmax = np.array(l_dev_softmax)
l_dev_softmax = l_dev_softmax.astype(np.float64)

l_eval_softmax_path = 'soft_result.csv'
lf = open(l_eval_softmax_path,'r')
csv_reader = csv.reader(lf)
l_eval_softmax =[]
for row in csv_reader:
    l_eval_softmax.append(row)
lf.close()
l_eval_softmax = np.array(l_eval_softmax)
l_eval_softmax = l_eval_softmax.astype(np.float64)

l_dev_pred = np.argmax(l_dev_softmax, 1)
l_eval_pred = np.argmax(l_eval_softmax, 1)

#
if mode == 'dev':
    fusion_dev_pred = fusion(l_dev_softmax, m_dev_softmax, 'dev')
    fusion_accu_list, fusion_all_accu = matrix(fusion_dev_pred)
    l_accu_list, l_all_accu = matrix(l_dev_pred)
    m_accu_list, m_all_accu = matrix(m_dev_pred)
    f = open(dev_result, 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(['Fusion: '] + fusion_accu_list)
    writer.writerow(['Fusion Ave: ', fusion_all_accu] )
    writer.writerow(['-------------------------------------------------------------'])
    writer.writerow(['l_result: '] + l_accu_list)
    writer.writerow(['l_result Ave: ', l_all_accu])
    writer.writerow(['-------------------------------------------------------------'])
    writer.writerow(['m_result: '] + m_accu_list)
    writer.writerow(['m_result Ave: ', m_all_accu])
    f.close()
else:
    fusion_eval_pred = fusion(l_eval_softmax, m_eval_softmax, 'eval')
    write_csv(fusion_csvName, fusion_eval_pred)
    write_csv(l_csvName,l_eval_pred)
    write_csv(m_csvName, m_eval_pred)








