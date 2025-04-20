import numpy as np
import os
import time
import cv2
from tqdm import tqdm
import numpy
def list_pictures(directory):
    image_list = []
    imgType_list = ['jpg','bmp','png','jpeg','rgb','tif','JPG']
    for root1,dirs_name,files_name in os.walk(directory):
        for i in files_name:
            file_name = os.path.join(root1, i)
            if file_name.split('.')[-1] in imgType_list:
                image_list.append(file_name)
            #print(file_name)
    return image_list


def read_image( path ):

    train_datas = []
    train_lables = []
    roi_path = list_pictures(path)

    i = 0
    class_list = []
    img_num = []
    for image_list in roi_path:  
        if 'original'  in image_list:
            continue
        class_name = '_'.join(image_list.split('/')[-4:-1])
        if class_name not in class_list:#如果该类还没有被加入，第一次加入该类
            class_list.append(class_name)
            i = i + 1
            image_label = i-1
            img_num.append(0)
        else:
            image_label = class_list.index(class_name)

        if image_label >= 0: #86个人，选取训练集
            if img_num[image_label] < 50:
                train_lables.append(image_label)
                train_datas.append(image_list)
                img_num[image_label] = img_num[image_label] + 1
                    
    return train_datas,train_lables


def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"       

    t0 = int(time.time())
    # path = '/media/shaohuikai/public1/ROI/test/user_roi416'  # path of dataset 
    path = '/media/shaohuikai/public1/ALL_image/Open/IR_test'
    total_data, label= read_image(path)
    print('-----------',len(total_data))  
    # ----------------------------------------------------------------------------------------------------------------------
    # ==> Network Initialization 
    # ModelPath =  '/media/shaohuikai/public1/ROI/test/consine_model.pb'
    ModelPath =  './model/240717/IR_MF_IR_ada_pretrained_512/RIR_240719_1_02306_1_1.onnx'
    net = cv2.dnn.readNetFromONNX(ModelPath)
    feature_dim = 128
    #--------------------------------------------------------------------------------------------------
    true_list = []
    false_list=[]   
    result_code = []
    source = []
    target = []
    #input_x = sess.graph.get_tensor_by_name('input:0')
    distance = 'consin' # Euclidean  consin

    for img_path in tqdm(total_data):
        img = cv2.imread(img_path) 
        img = img / 255
        if 'IR' in path:            
            img = img - 0.4648
            # img = img * 0.0078125
            img = img / 0.0641
        else:
            img[:,:,0] -= 0.6064
            img[:,:,0] /= 0.0599
            img[:,:,1] -= 0.4929
            img[:,:,1] /= 0.0614
            img[:,:,2] -= 0.4999
            img[:,:,2] /= 0.0668
        # img = np.reshape(img, (1, 112, 112, 3))
        img = img.astype(numpy.float32) 

        blob_img = cv2.dnn.blobFromImage(img, 1.0 , (112, 112), None, True, False)          #需要对应
        net.setInput(blob_img)
        # out_info = dnn_net.getUnconnectedOutLayersNames()  # this function need version 4.1.0 of opencv (or so).
        feature = net.forward()

        code = np.reshape(feature,[1,-1])
        
        result_code.append(code)

    result_code=np.reshape(result_code,[-1,feature_dim])  
   
    #-------------------------------------------------------------------------------------------------
    # used to obtain the DET curve and EER
    for i in tqdm(range(len(result_code))): #匹配
        # print(i)
        for j in range(i+1, len(result_code)):
            if label[i] == label[j]:
                if distance == 'consin':
                    true_list.append(np.dot(result_code[i],result_code[j]))
                else:
                    true_list.append(np.sqrt(np.sum(np.square(result_code[i] - result_code[j]))))
            #    print("true_list:np.dot(result_code[i],result_code[j]):",np.dot(result_code[i],result_code[j]))
            #   image_list.append([total_data[i], total_data[j]])
            else:
                if distance == 'consin':
                    false_list.append(np.dot(result_code[i],result_code[j]))
                else:
                    false_list.append(np.sqrt(np.sum(np.square(result_code[i] - result_code[j]))))
            #    print("false_list:np.dot(result_code[i],result_code[j]):",np.dot(result_code[i],result_code[j]))
    #-------------------------------------------------------------------------------------------------
    # used to get the threshold as different FAR
    print(max(label)+1)
    if distance == 'consin':
        false_list = sorted(false_list,reverse = True) #降序
    else:
        false_list = sorted(false_list,reverse = False) #升序
    #true_list = sorted(true_list) # 升序        
    total_true = len(true_list)
    total_false = len(false_list)
    FAR = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
    m = 0
    for far in FAR:
        if far * total_false < 1:
            break
        thr = false_list[int(far * total_false)]
        for num, true_ in enumerate(true_list):
            if distance == 'consin':
                if true_ < thr:
                    m = m +1
            else:
                if true_ > thr:
                    m = m +1
                # if far == 0.0001:
                #     print('bad case', image_list[num], true_)
            #else:
            #    break
        TAR = (total_true - m) / total_true
        m = 0
        print('@FAR={}, Thr={}, TAR={}'.format(far,thr,TAR))

    # np.savetxt("./true.txt",true_list,fmt="%f")
    # np.savetxt("./false.txt",false_list,fmt="%f")
    print('Done')

    t = int(time.time())
    print('Time elapsed: {}h {}m'.format((t - t0) // 3600, ((t - t0) % 3600) // 60))
  



if __name__ == '__main__':
    main()
