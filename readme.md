KVsystem.py为主要系统文件。
query接口实现了查询功能,不存在的key返回空。
seek接口实现查找功能，返回第一个大于等于key的键值对，支持不存在的key。
系统的配置能通过config.py实现
dataFS.py对数据文件进行操作。保证能在输入任意索引的情况下，都能返回到符合预期的值。保证了系统的正确性。
btree.py gbdt.py linear.py 为索引模型的具体实现
