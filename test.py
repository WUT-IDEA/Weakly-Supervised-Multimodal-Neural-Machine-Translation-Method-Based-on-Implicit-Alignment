import numpy as np
import codecs

# lista = [3,2,4,6,5,7,19,21,34,60,0,0,0,0,0]
# listb = sorted(lista)
# listb[2] = 10

# def random_change(list):
#     list_tmp = []
#     list_return = []
#     i = 0
#     while list[i] != 0 and i < 15:
#         if np.random.uniform(0, 1) < (1-0.3):
#             list_tmp.append(list[i])
#         i += 1

#     index_orig = [i for i in range(len(list_tmp))]
#     for i in range(len(index_orig)):
#         index_orig[i] += np.random.uniform(0, 3)
#     list_index = sorted(range(len(index_orig)), key=lambda k: index_orig[k])
#     for i in list_index:
#         list_return.append(list_tmp[i])
#     for i in range(0, 15 - len(list_index)):
#         list_return.append(0)
#     return list_return


# print(lista)
# print(random_change(lista))


# def load_img_data(in_path):
#     first_line = True
#     temp_list = []
#     with codecs.open(in_path, 'r', 'utf-8') as f_in:
#         for line in f_in:
#             if first_line:
#                 first_line = False
#                 continue
#             max_len_list = []
#             vec = [float(x) for x in line.strip().split(",")[1:]]
#             for i in range(50):
#                 max_len_list.append(vec)
#             temp_list.append(max_len_list)
#     data = np.array(temp_list, dtype=np.float)
#     return data


# print(load_img_data('/data1/home/aha12315/Data_Transformer_tf/data/VGG_image_features_tmp/test_features_en_512.csv'))


#################输出checkpoints_list################
# model = []
# for i in range(1, 113):
#     model.append("model_epoch_" + ("0" if i < 10 else "")+ str(i) + "_gs_" + str(i*906))

# print(list(reversed(model)))


#################修改checkpoint文件#################
# model = []
# for i in range(1, 113):
#     model.append("model_epoch_" + ("0" if i < 10 else "")+ str(i) + "_gs_" + str(i*906))

# with codecs.open("tangle01/checkpoint", "w", "utf-8") as fout:
#     writes = "model_checkpoint_path: \"" + model[1] + "\"\nall_model_checkpoint_paths: \"" + model[1] + "\"\n"
#     fout.write(writes)