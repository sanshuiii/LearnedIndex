import BasicFS

if __name__=="__main__":
    bs = BasicFS('sorted_demo_data')
    x, y = bs.gen_test_data()
    print(x)
    print(y)