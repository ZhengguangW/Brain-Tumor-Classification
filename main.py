from ReadImages import ReadImages

Factory=ReadImages("/Users/zhengguang/Desktop/OneDrive - University of Virginia/Desktop/CS 4774/Brain-Tumoer-Classification/archive")
Factory.Reading()
Factory.Split()
print("finished")
Factory.get_training()
Factory.get_testing()
Factory.get_validation()