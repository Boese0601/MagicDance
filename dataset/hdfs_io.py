from typing import IO, Any, List
import subprocess
import os
from contextlib import contextmanager
import json
import shutil

HADOOP_BIN = 'HADOOP_ROOT_LOGGER=ERROR,console /opt/tiger/yarn_deploy/hadoop/bin/hdfs'

@contextmanager  # type: ignore
def hopen(hdfs_path: str, mode: str = "r") -> IO[Any]:
    """
        打开一个 hdfs 文件, 用 contextmanager.

        Args:
            hfdfs_path (str): hdfs文件路径
            mode (str): 打开模式，支持 ["r", "w", "wa"]
    """
    pipe = None
    if mode.startswith("r"):
        pipe = subprocess.Popen(
            "{} dfs -text {}".format(HADOOP_BIN, hdfs_path), shell=True, stdout=subprocess.PIPE)
        yield pipe.stdout
        pipe.stdout.close()  # type: ignore
        pipe.wait()
        return
    if mode == "wa" or mode == "a":
        pipe = subprocess.Popen(
            "{} dfs -appendToFile - {}".format(HADOOP_BIN, hdfs_path), shell=True, stdin=subprocess.PIPE)
        yield pipe.stdin
        pipe.stdin.close()  # type: ignore
        pipe.wait()
        return
    if mode.startswith("w"):
        pipe = subprocess.Popen(
            "{} dfs -put -f - {}".format(HADOOP_BIN, hdfs_path), shell=True, stdin=subprocess.PIPE)
        yield pipe.stdin
        pipe.stdin.close()  # type: ignore
        pipe.wait()
        return
    raise RuntimeError("unsupported io mode: {}".format(mode))


def hlist_files(folders: List[str]) -> List[str]:
    """
        罗列一些 hdfs 路径下的文件。

        Args:
            folders (List): hdfs文件路径的list
        Returns:
            一个list of hdfs 路径
    """
    files = []
    for folder in folders:
        if folder.startswith('hdfs'):
            pipe = subprocess.Popen("{} dfs -ls {}".format(HADOOP_BIN, folder), shell=True,
                                    stdout=subprocess.PIPE)
            # output, _ = pipe.communicate()
            for line in pipe.stdout:  # type: ignore
                line = line.strip()
                # drwxr-xr-x   - user group  4 file
                if len(line.split()) < 5:
                    continue
                files.append(line.split()[-1].decode("utf8"))
                #print( line.split()[-1].decode("utf8") )
            pipe.stdout.close()  # type: ignore
            pipe.wait()
        else:
            if os.path.isdir(folder):
                files.extend([os.path.join(folder, d) for d in os.listdir(folder)])
            elif os.path.isfile(folder):
                files.append(folder)
            else:
                #print('Path {} is invalid'.format(folder))
                continue
    return files

def hexists(file_path: str) -> bool:
    """ hdfs capable to check whether a file_path is exists """
    if file_path.startswith('hdfs'):
        return os.system("{} dfs -test -e {}".format(HADOOP_BIN, file_path)) == 0
    return os.path.exists(file_path)

def hisdir(file_path: str) -> bool:
    """ hdfs capable to check whether a file_path is a dir """
    if file_path.startswith('hdfs'):
        flag1 = os.system("{} dfs -test -e {}".format(HADOOP_BIN, file_path))  # 0:路径存在
        flag2 = os.system("{} dfs -test -f {}".format(HADOOP_BIN, file_path))  # 0:是文件 1:不是文件
        flag = ((flag1 == 0) and (flag2 == 1))
        return flag
    return os.path.isdir(file_path)

def hmkdir(file_path: str) -> bool:
    """ hdfs mkdir """
    if file_path.startswith('hdfs'):
        os.system("{} dfs -mkdir -p {}".format(HADOOP_BIN, file_path))
    else:
        os.makedirs(file_path, exist_ok=True)
    return True

def hcopy(from_path: str, to_path: str) -> bool:
    """ hdfs copy """
    if to_path.startswith("hdfs"):
        if from_path.startswith("hdfs"):
            os.system("{} dfs -cp -f {} {}".format(HADOOP_BIN, from_path, to_path))
        else:
            os.system("{} dfs -copyFromLocal -f {} {}".format(HADOOP_BIN, from_path, to_path))
    else:
        if from_path.startswith("hdfs"):
            os.system("{} dfs -text {} > {}".format(HADOOP_BIN, from_path, to_path))
        else:
            shutil.copy(from_path, to_path)
    return True

if __name__ == "__main__":
    data_path = 'hdfs://haruna/home/byte_labcv_gan/public/multimodal_all/tusou2E_clean'
    files = hlist_files(data_path.split(','))
    files = [f for f in files if f.find('_SUCCESS') < 0]
    for filepath in files:
        with hopen(filepath, 'r') as reader:
            for line in reader:
                data_item = json.loads(line)