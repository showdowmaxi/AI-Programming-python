import model_f
import argparse
import json
import torch
#set out associate predict commands:

def command_line():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type = str, help = 'A path of an image')
    parser.add_argument('checkpoint',type =str,action ='store',help = 'Model checkpoint file for prediction')
    parser.add_argument('--top_k', type = int, default = 3, help='Top rate of predict flower')
    parser.add_argument('--category_names', type = str, action = 'store', help='Top rate of predict flower')
    parser.add_argument('--gpu', dest='gpu',action = 'store_false', help = 'Set to gpu mode')
    
    
    return  parser.parse_args()

    
def main():
    #get the input from commd
    args = command_line()
    use_gpu = torch.cuda.is_available() and args.gpu
    
    #get model
    model = model_f.load_check_point(args.checkpoint)
    #return the given top rate
    pro, classes = model_f.predict(args.input, model, args.top_k, use_gpu)
    if args.top_k:
        print(classes)
        print(pro)
        
    #get flower name
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    #get flower name
    name = model_f.get_name(classes,cat_to_name)
    print(name)

if __name__ == "__main__":
    main()