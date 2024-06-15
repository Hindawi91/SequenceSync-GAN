import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_loader
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from tensorboardX import SummaryWriter
import argparse
import datetime
# from sklearn.metrics import accuracy_score


# domain_folder = "A"





def classification_loss(logit, target):
    """Compute binary or softmax cross entropy loss."""

    return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)


def accuracy_fn(pred,true):
    _, argmax = torch.max(pred, 1)
    accuracy = (true == argmax.squeeze()).float().mean()

    return accuracy
  
def label2onehot(labels, dim = 2):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim, device=labels.device)
        out[np.arange(batch_size), labels.long()] = 1
        return out

class Temporal_Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=256, conv_dim=64, c_dim=2, repeat_num=6):
        super(Temporal_Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h) # real or fake
        out_cls = self.conv2(h) # what class
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))


# def process_dataiter_output(images, labels, file_names, frame_numbers):
#     if frame_numbers[0]<frame_numbers[1]<frame_numbers[2]:

#         # print("Batch in sequence")
#         sequence_label = torch.tensor([[1]])
#     else:
#         # print("Batch NOT in sequence")
#         sequence_label = torch.tensor([[0]])

#     seq_image = images.squeeze(1)
#     seq_image = seq_image.unsqueeze(0)
#     seq_image = seq_image.to(device)
#     sequence_label = sequence_label.to(device)
#     sequence_label = sequence_label.float()
#     sequence_label_hot = label2onehot(sequence_label)

#     return seq_image,sequence_label,sequence_label_hot


# def create_actual_batch(actual_batch_size,data_iter):

#     epoch_end = False

#     for i in range(actual_batch_size):

#         try:
#             images, labels, file_names, frame_numbers = next(data_iter)
#             seq_image, sequence_label,sequence_label_hot = process_dataiter_output(images, labels, file_names, frame_numbers)

#             if i == 0:
#                 batch_seq_imgs = seq_image.clone()
#                 batch_seq_labels = sequence_label.clone()
#                 batch_seq_labels_hot = sequence_label_hot.clone()
#             else:
#                 batch_seq_imgs = torch.cat((batch_seq_imgs,seq_image), dim =0)
#                 batch_seq_labels = torch.cat((batch_seq_labels,sequence_label), dim =0)
#                 batch_seq_labels_hot = torch.cat((batch_seq_labels_hot,sequence_label_hot), dim =0)

#         except:
#             epoch_end = True
#             break

#     return batch_seq_imgs,batch_seq_labels,batch_seq_labels_hot,epoch_end




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Your program description')
    parser.add_argument('--domain', type=str, help='Enter the domain name')
    parser.add_argument('--batch_size', type=int, help='Enter the batch_size',default = 16)
    parser.add_argument('--device', type=str, help='Enter the batch_size',default = 'cuda:3')
    args = parser.parse_args()

    domain_folder = args.domain
    batch_size = args.batch_size
    device = args.device

    device = torch.device(device if torch.cuda.is_available() else 'cpu')


    train_data_loader = get_loader(image_dir ="./data", image_size = 256, 
                   batch_size=batch_size, dataset='Boiling', mode='train', num_workers=1,domain = domain_folder)

    val_data_loader = get_loader(image_dir ="./data", image_size = 256, 
                   batch_size=batch_size, dataset='Boiling', mode='val', num_workers=1,domain = domain_folder)

    test_data_loader = get_loader(image_dir ="./data", image_size = 256, 
                   batch_size=batch_size, dataset='Boiling', mode='test', num_workers=1,domain = domain_folder)



    Temporal_Discriminator = Temporal_Discriminator().to(device)
    # criterion = nn.BCELoss()
    optimizer = optim.Adam(Temporal_Discriminator.parameters(), lr=0.001)


    date_time = datetime.datetime.now()
    date_time = date_time.strftime("%d_%m_%Y")

    # TensorBoard writer
    writer = SummaryWriter(f"./logs/Domain_{domain_folder}_({date_time})")


    # Training the model


    best_val_loss = float('inf')
    best_model = None

    iterations = 300000


    for iteration in range(iterations):

        Temporal_Discriminator.train()

        train_iter = iter(train_data_loader)
        images, labels = next(train_iter)
        images, labels = images.to(device), labels.to(device)

        labels_hot = label2onehot(labels)

        # Forward pass
        out_src, out_cls = Temporal_Discriminator(images)
        loss = classification_loss(out_cls, labels_hot)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute accuracy
        accuracy = accuracy_fn(out_cls,labels)

        # Print and write to TensorBoard
        if (iteration + 1) % 10 == 0:
            print(f'iteration [{iteration + 1}/{iterations}], Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.2f}')
            writer.add_scalar('training loss', loss.item(), iteration+1)
            writer.add_scalar('training accuracy', accuracy.item(), iteration+1)

        if (iteration + 1) % 1000 == 0:

            Temporal_Discriminator.eval()

            with torch.no_grad():
                total_val_loss = 0
                total_val_samples = 0
                val_iter = iter(val_data_loader)

                for val_images, val_labels in val_iter:

                    val_images, val_labels = val_images.to(device), val_labels.to(device)

                    val_labels_hot = label2onehot(val_labels)


                    out_src, out_cls = Temporal_Discriminator(val_images)
                    batch_val_loss = classification_loss(out_cls, val_labels_hot)

                total_val_loss += batch_val_loss.item() 
                total_val_samples += val_labels.size(0)

            average_val_loss = total_val_loss / total_val_samples
            print(f'Validation Loss: {average_val_loss:.4f}')
            writer.add_scalar('validation loss', average_val_loss, iteration)

            if average_val_loss < best_val_loss:
                best_val_loss = average_val_loss
                torch.save(Temporal_Discriminator.state_dict(), f'Temporal_model_domain{domain_folder}.pth')
                print (f"Model Saved as Temporal_model_domain{domain_folder}.pth")

        # print(f"{epoch =},{average_val_loss =}")

    writer.close()

    # for epoch in range(epochs):

    #     if new_epoch:
    #         train_iter = iter(train_data_loader)
    #         images, labels = next(data_iter)
    #         new_epoch = False
    #         iteration = 0

    #     Temporal_Discriminator.train()

    #     while new_epoch == False:
    #         batch_seq_imgs,batch_seq_labels,batch_seq_labels_hot,epoch_end = create_actual_batch(actual_batch_size=actual_batch_size,data_iter=train_iter)
    #         new_epoch = epoch_end
    #         iteration += 1

    #         # torchvision.utils.save_image(batch_seq_imgs, f'batch_seq_imgs.png', nrow=2, padding=2, normalize=True)

            
    #         # Forward pass
    #         out_src, out_cls = Temporal_Discriminator(batch_seq_imgs)
    #         loss = classification_loss(out_cls, batch_seq_labels_hot)

    #         # Backward and optimize
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         # Compute accuracy
    #         accuracy = accuracy_fn(out_cls,batch_seq_labels)

    #         # print(f"{iteration=}")

    #         # Print and write to TensorBoard
    #         if (iteration + 1) % 10 == 0:
    #             print(f'epoch [{epoch + 1}/{epochs}], iteration [{iteration + 1}/{iters_per_epoch}], Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.2f}')
    #             writer.add_scalar('training loss', loss.item(), epoch * iters_per_epoch + iteration)
    #             writer.add_scalar('accuracy', accuracy.item(), epoch * iters_per_epoch + iteration)


    #         # Validation ---------------------------problem with batches---------------------------
    #     Temporal_Discriminator.eval()

    #     with torch.no_grad():
    #         total_val_loss = 0
    #         total_val_samples = 0
    #         data_end = False
    #         val_iter = iter(val_data_loader)

    #         while data_end == False:
    #             batch_seq_imgs,batch_seq_labels,batch_seq_labels_hot,data_end = create_actual_batch(actual_batch_size=actual_batch_size,data_iter=val_iter)

    #             out_src, out_cls = Temporal_Discriminator(batch_seq_imgs)
    #             batch_val_loss = classification_loss(out_cls, batch_seq_labels_hot)

    #             total_val_loss += batch_val_loss.item() 
    #             total_val_samples += batch_seq_labels.size(0)

    #         average_val_loss = total_val_loss / total_val_samples
    #         print(f'Validation Loss: {average_val_loss:.4f}')
    #         writer.add_scalar('validation loss', average_val_loss, epoch)

    #         if average_val_loss < best_val_loss:
    #             best_val_loss = average_val_loss
    #             torch.save(Temporal_Discriminator.state_dict(), f'Temporal_model_domain{domain_folder}.pth')
    #             print (f"Model Saved as Temporal_model_domain{domain_folder}.pth")

    #     # print(f"{epoch =},{average_val_loss =}")

    # writer.close()





        

        # print(loss)

        # val_loss = classification_loss(val_out_cls, val_sequence_label)

        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     best_model = Temporal_Discriminator.state_dict()
        #     torch.save(best_model, 'best_model_checkpoint.pth')
        #     print("checkpoint model saved")

        # if iteration%100 == 0:
        #     print(f"{iteration =},{loss.item() =},{val_loss.item()=}")
        # loss.backward()
        # optimizer.step()

# num_epochs = 5000

# for epoch in range(num_epochs):
#     print(f"{epoch =}")
#     for batch in data_loader:
#         images, labels, file_names, frame_numbers = batch

#         if frame_numbers[0]<frame_numbers[1]<frame_numbers[2]:

#             # print("Batch in sequence")
#             sequence_label = torch.tensor([[1]])
#         else:
#             # print("Batch NOT in sequence")
#             sequence_label = torch.tensor([[0]])

#         seq_image = images.squeeze(1)
#         seq_image = seq_image.unsqueeze(0)
#         seq_image = seq_image.to(device)
#         sequence_label = sequence_label.to(device)

#         # torchvision.utils.save_image(seq_image, f"./seq_image_epoch_{epoch}.png")    
#         optimizer.zero_grad()

#         out_src, out_cls = discriminator(seq_image)
#         sequence_label = sequence_label.float()


#         sequence_label = label2onehot(sequence_label)

#         loss = classification_loss(out_cls, sequence_label)
#         loss.backward()
#         optimizer.step()


