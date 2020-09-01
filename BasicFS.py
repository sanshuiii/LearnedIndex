import os
import logging
import pandas as pd
import numpy as np


class DataProvider():
    ITEMLEN = 61
    MAXLEN = 30

    def __init__(self, filename='KVFIle'):
        if not os.path.exists(filename):
            with open(filename, 'w', encoding='utf-8'):
                pass
        self.file = open(filename, 'r+t', encoding='utf-8')
        self.file.seek(0)
        buf=self.file.read(self.ITEMLEN)
        self.first_item=buf.find('\n')

    def _splitKV(self,buf:str)->(bool,str):
        if buf.count(' ') != 1:
            logging.warning("not valid KV pair:{}".format(buf))
            return False, ''
        return True, buf.split(' ')[1]

    def query_value(self, offset, fuzzy=True) -> (bool, str):
        """

        :param offset:
        :param fuzzy: 是否是模糊模式，精准模式认为offset指向KV的头
        :return:
        """
        #第一个KV对的特殊处理
        if offset<self.first_item:
            if offset!=0:
                logging.warning("offset:{}\t is not a head of KV pair,and this query is exact mode".format(offset))
                return False, ''
            self.file.seek(0)
            buf = self.file.read(self.ITEMLEN)
            if not fuzzy and offset!=0:
                return False,''
            return self._splitKV(buf[:self.first_item])

        self.file.seek(max(offset - self.ITEMLEN,0))
        buf = self.file.read(self.ITEMLEN * 2 + 2)

        if self.ITEMLEN*2+2-len(buf)>0:#结束部分
            if len(buf)<self.ITEMLEN:
                logging.warning("offset:{} too big".format(offset))
                return False, ''
            else:
                buf = buf.strip()
                return self._splitKV(buf[buf.rfind("\n"):])


        pos=self.ITEMLEN
        if not fuzzy and buf[pos - 1] != '\n':  # 精准模式下没找到
            logging.warning("offset:{}\t is not a head of KV pair,and this query is exact mode".format(offset))
            return False, ''
        if buf[pos-1] == '\n':  # 准确找到了
            tail=buf[pos:].find('\n')+pos
            return self._splitKV(buf[pos:tail])
        # 确定offset所在的行

        head,tail = -1,-1
        for i in range(pos,-1,-1):
            if buf[i]=='\n':
                head=i+1
                break
        for i in range(pos+1,len(buf)):
            if buf[i]=='\n':
                tail=i
                break
        if tail>head:
            return self._splitKV(buf[head:tail])
        else:
            return False,''

    def statistics(self, fill_zero=True):
        """

        :param fill_zero: 不满30位是否补充后置0
        :return:
        """
        self.file.seek(0)
        df = pd.DataFrame(None, index=[i for i in range(self.MAXLEN)],
                          columns=[str(i) for i in range(10)] + [chr(i) for i in range(ord('a'), ord('g'))],
                          dtype=np.int)
        df = df.fillna(0)
        len_freq = {}
        distribute = {}
        offset = 0
        for line in self.file.readlines():
            # print(offset)
            offset += len(line)
            k = line[:line.find(' ')]
            len_freq[len(k)] = len_freq.get(len(k), 0) + 1
            if fill_zero and len(k) < self.MAXLEN:
                k = k + "0" * (self.MAXLEN - len(k))
            v = self._str2base10(k, False)
            for idx, b in enumerate(k):
                df[b][idx] += 1
            # distribute[v]=distribute.get(v,0)+1
            distribute[offset] = v / 1000000000
        with open('distribute.csv', 'w') as f:
            for k, v in distribute.items():
                f.write("{},{}\n".format(k, v))
        df.to_csv('freq.csv')

    def gen_offset(self) -> dict:
        """

        :return: {key:offset} the type of key is str and the type of offset is int
        """
        offset = 0
        self.file.seek(0)
        res = {}
        for line in self.file.readlines():
            # print(offset)
            offset += len(line)
            k = line[:line.find(' ')]
            res[k] = offset
        return res

    def _str2base10(self, str, fill=True):
        if fill and len(str) < self.MAXLEN:
            str = str + "0" * (self.MAXLEN - len(str))
        return int(str, 16)

    def gen_test_data(self, isheader=True) -> (list, list):
        """
        :param isheader:是否返回记录头 todo
        :return:X是字符串  Y是整数
        """
        offset = 0
        self.file.seek(0)
        X = []
        Y = []
        for line in self.file.readlines():
            offset += len(line) / 2 if isheader else len(line)
            k = line[:line.find(' ')]
            X.append(self._str2base10(k))
            Y.append(offset)
        return X, Y


if __name__ == '__main__':
    bs = DataProvider('sorted_demo_data')
    bs.file.seek(0,2)
    print(bs.file.tell())
    for i in range(36054-5,36054+100):
        v = bs.query_value(i,fuzzy=True)
        print(v)
