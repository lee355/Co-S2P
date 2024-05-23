# -*- coding: gbk -*-
import paramiko
import os
import subprocess
import pexpect


local_folder = './dataset/imagenet2012/fed_data/new_client_16non-iid_1.5'


servers = [
    {'hostname': '10.102.32.203', 'username': 'xugw', 'password': 'iic123456', 'remote_folder': '/home/xugw/fed_data/'},
    {'hostname': '10.102.32.203', 'username': 'xugw', 'password': 'iic123456', 'remote_folder': '/home/xugw/fed_data/'},
    {'hostname': '101.76.210.28', 'username': 'nvidia', 'password': 'nvidia', 'remote_folder': '/home/nvidia/fed_data/'},
    {'hostname': '101.76.219.170', 'username': 'nvidia', 'password': 'nvidia', 'remote_folder': '/home/nvidia/fed_data/'},
    {'hostname': '101.76.220.29', 'username': 'nvidia', 'password': 'nvidia', 'remote_folder': '/home/nvidia/fed_data/'},
    {'hostname': '101.76.214.45', 'username': 'nvidia', 'password': 'nvidia', 'remote_folder': '/home/nvidia/fed_data/'},
    {'hostname': '101.76.212.157', 'username': 'nvidia', 'password': 'nvidia', 'remote_folder': '/home/nvidia/fed_data/'},
    {'hostname': '101.76.210.134', 'username': 'nvidia', 'password': 'nvidia', 'remote_folder': '/home/nvidia/fed_data/'},
    {'hostname': '101.76.215.73', 'username': 'nvidia', 'password': 'nvidia', 'remote_folder': '/home/nvidia/fed_data/'},
    {'hostname': '101.76.211.61', 'username': 'nvidia', 'password': 'nvidia', 'remote_folder': '/home/nvidia/fed_data/'},
    {'hostname': '101.76.220.69', 'username': 'nvidia', 'password': 'nvidia', 'remote_folder': '/home/nvidia/fed_data/'},
    {'hostname': '101.76.218.162', 'username': 'nvidia', 'password': 'nvidia', 'remote_folder': '/home/nvidia/fed_data/'},
    {'hostname': '101.76.212.180', 'username': 'nvidia', 'password': 'nvidia', 'remote_folder': '/home/nvidia/fed_data/'},
    {'hostname': '101.76.216.237', 'username': 'nvidia', 'password': 'nvidia', 'remote_folder': '/home/nvidia/fed_data/'},
    {'hostname': '101.76.211.239', 'username': 'nvidia', 'password': 'nvidia', 'remote_folder': '/home/nvidia/fed_data/'},
    {'hostname': '101.76.220.159', 'username': 'nvidia', 'password': 'nvidia', 'remote_folder': '/home/nvidia/fed_data/'}
]

# 遍历每个远程服务器
for i in range(len(servers)):
    print("Start transferring client" + str(i))
    servers[i]['remote_folder'] += 'new_client_16non-iid_1.5'
    sub_local_folder = local_folder + "/client_" + str(i)
    # 建立SSH连接
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(servers[i]['hostname'], username=servers[i]['username'], password=servers[i]['password'])

    # 创建SFTP客户端
    sftp = ssh.open_sftp()

    try:
        sftp.chdir(servers[i]['remote_folder'])
    except IOError:
        sftp.mkdir(servers[i]['remote_folder'])
        
    # 递归传输子文件夹
    for root, dirs, files in os.walk(sub_local_folder):
        remote_dir = os.path.join(servers[i]['remote_folder'], os.path.relpath(root, local_folder))
        try:
            sftp.mkdir(remote_dir)
        except IOError:
            pass
        for file in files:
            sftp.put(os.path.join(root, file), os.path.join(remote_dir, file))

    # 关闭SFTP客户端和SSH连接
    sftp.close()
    ssh.close()
