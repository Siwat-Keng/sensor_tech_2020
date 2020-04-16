import Mapping

#Define Resolution
m = 1.0
dm = 0.1
cm = 0.01
mm = 0.001

if __name__ == '__main__':
    # Mapping.collect_map(repeat=400) #save .npy map 
    Mapping.convert_map(resulution=cm) #convert all .npy to .png resolution = 1cm