import model_f
import argparse
import torch
#set out associate train commands:

def command_line():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type = str, help = 'Directory of flower')
    parser.add_argument('--save_dir', type = str, default = 'checkpoint.pth', action = 'store',help='Save checkpoint')  
    model_type={'vgg16','vgg19'}
    parser.add_argument('--arch',dest='arch',default='vgg16', action ='store',choices = model_type ,help='Model architecture for training')
    parser.add_argument('--learning_rate', type = float, default = 0.001,help = 'Learning rate of model')
    parser.add_argument('--hidden_units', type = int, default = 2010 ,help = 'Number of hidden unit')
    parser.add_argument('--epochs', type = int, default = 3,help = 'Number of epochs of trainning model')
    parser.add_argument('--gpu', dest='gpu',action ='store_false', help = 'Set to gpu mode')
    
    result = parser.parse_args()
    return result
    
def main():
    
    #get the input from commd
    args = command_line()
        
    #get data
    dataloaders, class_to_idx = model_f.get_data(args.data_dir)
    trainloaders,validloaders = dataloaders['trainloaders'],dataloaders['validloaders']
    
    #set gpu
    use_gpu = torch.cuda.is_available() and args.gpu
    
    #pretrain model and get
    model,criterion,optimizer = model_f.set_model(args.arch,args.hidden_units,args.learning_rate)
    model_f.pretrain_model(args.epochs,use_gpu,trainloaders,validloaders,model,criterion,optimizer)
    
    #set path
    model_f.set_checkpoint(model,optimizer,args.epochs,args.learning_rate,args.hidden_units,args.save_dir,class_to_idx,args.arch)
       
if __name__ == "__main__":
    main()