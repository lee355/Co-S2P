# -*- coding: gbk -*-
import paramiko
import os
import subprocess
import pexpect

# 本地文件夹路径
local_folder = '/home/ly/code/src/code and baselines/dataset/imagenet2012/imagenet200/fed_data/client_8non-iid_0.5'

# 远程服务器信息
servers = [
    {'hostname': '101.76.222.144', 'username': 'nvidia', 'password': 'nvidia', 'remote_folder': '/home/nvidia/fed_data'},
    {'hostname': '101.76.221.47', 'username': 'nvidia', 'password': 'nvidia', 'remote_folder': '/home/nvidia/fed_data'},
    {'hostname': '101.76.214.25', 'username': 'nvidia', 'password': 'nvidia', 'remote_folder': '/home/nvidia/fed_data'},
    {'hostname': '101.76.216.223', 'username': 'nvidia', 'password': 'nvidia', 'remote_folder': '/home/nvidia/fed_data'},
    {'hostname': '101.76.209.143', 'username': 'nvidia', 'password': 'nvidia', 'remote_folder': '/home/nvidia/fed_data'},
    {'hostname': '101.76.219.195', 'username': 'nvidia', 'password': 'nvidia', 'remote_folder': '/home/nvidia/fed_data'},
    {'hostname': '101.76.214.47', 'username': 'nvidia', 'password': 'nvidia', 'remote_folder': '/home/nvidia/fed_data'}
]

# 遍历每个远程服务器
for i in range(len(servers)):
    print("Start transferring client" + str(i))
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
    #tmp = servers[i]['remote_folder']
    servers[i]['remote_folder'] += '/client_8non-iid_0.5'
    sub_local_folder = local_folder + "/client_" + str(i)

    
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