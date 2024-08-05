def write_list_to_file(data_list,path):
    print('save list at',path)
    file = open(path,'w')
    for (img_path,label) in data_list:
        file.write( img_path+' '+str(label)+'\n')
    file.close()

def read_list_from_file(path):
    file = open(path,'r')
    data_list= []
    for line in file:
        img_path,label = line.split()
        data_list.append( ( img_path , int(label) ) )
    file.close()
    return data_list

def split_dataset(list_dir):
    data_list = read_list_from_file(list_dir)
    import random
    random.seed(0)
    random.shuffle(data_list)
    test_list = data_list[:500]
    train_list = data_list[500:]
    base_dir = list_dir.split('.')[0]
    write_list_to_file( data_list,base_dir+'_all.txt')
    write_list_to_file( test_list,base_dir+'_test.txt')
    write_list_to_file( train_list,base_dir+'_train.txt')