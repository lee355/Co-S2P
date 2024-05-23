import jtop
import os

class device_property:
    device_module_64_32 = "AGX"
    device_memory_64 = "64"
    device_module_16_8 = "Xavier"
    device_memory_16 = "16"
    batchsize_64g = 320
    batchsize_32g = 256
    batchsize_16g = 192
    batchsize_8g = 24
    modelrate_64g = 0.75
    modelrate_32g = 0.75
    modelrate_16g = 0.5
    modelrate_8g = 0.25
    """ local_epoch_64g = 3
    local_epoch_32g = 3
    local_epoch_16g = 8
    local_epoch_8g = 8 """



def get_local_batch_size():
    tmp=jtop.jtop()
    tmp.start()
    module=tmp.board['hardware']['Module']
    tmp.close()
    if device_property.device_module_64_32 in module :
        if device_property.device_memory_64 in module:
            return device_property.batchsize_64g
        else :
            return device_property.batchsize_32g
    elif device_property.device_module_16_8 in module:
        if device_property.device_memory_16 in module :
            return device_property.batchsize_16g
        else :
            return device_property.batchsize_8g
    else :
        return 0

def get_local_model_rate():
    tmp=jtop.jtop()
    tmp.start()
    module=tmp.board['hardware']['Module']
    tmp.close()
    if device_property.device_module_64_32 in module :
        if device_property.device_memory_64 in module:
            return device_property.modelrate_64g
        else :
            return device_property.modelrate_32g
    elif device_property.device_module_16_8 in module:
        if device_property.device_memory_16 in module :
            return device_property.modelrate_16g
        else :
            return device_property.modelrate_8g
    else :
        return 0

def get_local_epoch():
    tmp=jtop.jtop()
    tmp.start()
    module=tmp.board['hardware']['Module']
    tmp.close()
    if device_property.device_module_64_32 in module :
        if device_property.device_memory_64 in module:
            return device_property.local_epoch_64g
        else :
            return device_property.local_epoch_32g
    elif device_property.device_module_16_8 in module:
        if device_property.device_memory_16 in module :
            return device_property.local_epoch_16g
        else :
            return device_property.local_epoch_8g
    else :
        return 0

def get_data_path():
    root_path='/home/nvidia/fed_data/client_7non-iid_1.5'
    if os.path.exists(root_path) is False:
        root_path='/home/***'
    dir=os.listdir(root_path)
    client=None
    for ddir in dir:
        if 'client' in ddir:
            client=ddir
            break
    root_path=root_path+'/'+client
    return root_path       