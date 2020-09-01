import os
import logging
import pandas as pd
import numpy as np

class DataProvider():

    ITEMLEN=61
    MAXLEN=30

    def __init__(self, filename='KVFIle'):
        if not os.path.exists(filename):
            with open(filename, 'w', encoding='utf-8'):
                pass
        self.file=open(filename, 'r+t', encoding='utf-8')

    def query_value(self, offset, fuzzy=True) -> (bool,str):
        self.file.seek(offset-self.ITEMLEN)
        buf = self.file.read(self.ITEMLEN*2+2)
        if not fuzzy and buf[0]!='\n':#精准模式下没找到
            logging.warning("offset:{}\t is not a head of KV pair,and this query is exact mode".format(offset))
            return False,''
        if buf[0]=='\n':#准确找到了
            if buf.find(' ')<=0:
                logging.warning("not valid KV pair in offset:{}".format(offset))
                return False,''
            begin=buf.find(' ')
            end=buf.rfind(' ') if buf.count(' ')>1 else -1
            return True,buf[begin:end].strip()
        #确定offset所在的行
        first_blank=buf[:offset-1].rfind('\n')
        second_blank=buf[offset-1:].find('\n')
        if first_blank>=0 and second_blank>=0:
            return True,buf[first_blank+1:second_blank].split(' ')[1]
        else:
            return False,''

    def statistics(self):
        self.file.seek(0)
        df=pd.DataFrame(None,index=[i for i in range(30)],
                        columns=[str(i) for i in range(10)]+[chr(i) for i in range(ord('a'),ord('g'))],dtype=np.int)
        df=df.fillna(0)
        #freq=[{}for i in range(30)]
        len_freq={}
        distribute={}
        offset=0
        for line in self.file.readlines():
            #print(offset)
            offset+=len(line)
            k=line[:line.find(' ')]
            len_freq[len(k)]=len_freq.get(len(k),0)+1
            if len(k)<30:
                k=k+"0"*(30-len(k))
            v=0
            for idx,b in enumerate(k):
                df[b][idx] += 1
                #freq[idx][b]=freq[idx].get('b',0)+1
                v+=eval("0x"+b)*(16**idx)/1000000000
            distribute[v]=distribute.get(v,0)+1
        with open('distribute.csv','w') as f:
            for k,v in distribute.items():
                f.write("{},{}\n".format(k,v))
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
            res[k]=offset
        return res

    def _str2base10(self, str):
        if len(str) < self.MAXLEN:
            str = str + "0" * (self.MAXLEN - len(str))
        return int(str,16)

    def gen_test_data(self,isheader=True)->(list,list):
        """
        :param isheader:是否返回记录头 todo
        :return:X是字符串  Y是整数
        """
        offset = 0
        self.file.seek(0)
        X = []
        Y = []
        for line in self.file.readlines():
            offset += len(line)
            k = line[:line.find(' ')]
            X.append(self._str2base10(k))
            Y.append(offset)
        return (X, Y)


if __name__ == '__main__':
    bs=DataProvider('sorted_demo_data')
    x,y = bs.gen_test_data()
    print(y)
