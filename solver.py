from model import Generator
from model import Temporal_Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
from sklearn.cluster import KMeans



class Solver(object):
    """Solver for training and testing Fixed-Point GAN."""

    # def __init__(self, data_loader, config):
    def __init__(self, data_loader_A, data_loader_B, config):
        """Initialize configurations."""

        self.data_loader_A = data_loader_A
        self.data_loader_B = data_loader_B
        self.direction = config.direction

        # Model configurations.
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp
        self.lambda_id = config.lambda_id
        self.lambda_TD = config.lambda_TD


        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.td_lr = config.td_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()



    def build_model(self):
        """Create a generator and a discriminator."""

        self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
        self.D = Temporal_Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num)    

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])

        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')

        self.G.to(self.device)
        self.D.to(self.device)


    def recreate_image(self, codebook, labels, w, h):
        """Recreate the (compressed) image from the code book & labels"""
        d = codebook.shape[1]
        image = np.zeros((w, h, d))
        label_idx = 0
        for i in range(w):
            for j in range(h):
                image[i][j] = codebook[labels[label_idx]]
                label_idx += 1
        return image

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))

        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out


    def create_labels(self, c_org):
        """Generate target domain labels for debugging and testing."""
        c_trg_list = []
        c_trg = c_org.clone()
        c_trg[:, 0] = (c_trg[:, 0] == 0)  # Reverse attribute value.
        c_trg_list.append(c_trg.to(self.device))
        return c_trg_list


    def classification_loss(self,logit, target):

        """Compute binary or softmax cross entropy loss."""

        return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)

    def split_images_by_label(self,x_fake,seq_label, label_trg):
        # Convert label_trg to a boolean mask
        # mask = label_trg.bool()
        mask = label_trg.squeeze().bool()

        # Split x_fake tensor based on the mask
        x_fake_A = x_fake[~mask]
        x_fake_B = x_fake[mask]

        seq_label_A = seq_label[~mask]
        seq_label_B = seq_label[mask]

        return x_fake_A, x_fake_B, seq_label_A, seq_label_B

    def train(self):
        """Train Fixed-Point GAN within a single dataset."""

        data_loader_A = self.data_loader_A
        data_loader_B = self.data_loader_B

        data_iter_A = iter(data_loader_A)
        data_iter_B = iter(data_loader_B)

        x_fixed_A, c_org_A, seq_label_A,file_names_A = next(data_iter_A)
        x_fixed_B, c_org_B, seq_label_B,file_names_B = next(data_iter_B)
        x_fixed = torch.cat((x_fixed_A, x_fixed_B), dim=0)
        x_fixed = x_fixed.to(self.device)
        c_org = torch.cat((c_org_A, c_org_B), dim=0)
        c_fixed_list = self.create_labels(c_org)


        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr
        td_lr = self.td_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()

        pre_CHF_count_A = 0
        CHF_count_A = 0

        pre_CHF_count_B = 0
        CHF_count_B = 0
        
        for i in range(start_iters, self.num_iters):
            # print(f"batch {i}")

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #


            try:
                x_real_A, label_org_A, seq_label_A,file_names_A = next(data_iter_A)
            except:
                data_iter_A = iter(data_loader_A)
                x_real_A, label_org_A, seq_label_A,file_names_A = next(data_iter_A)

            try:
                x_real_B, label_org_B, seq_label_B,file_names_B = next(data_iter_B)
            except:
                data_iter_B = iter(data_loader_B)
                x_real_B, label_org_B, seq_label_B,file_names_B = next(data_iter_B)

            x_real = torch.cat((x_real_A, x_real_B), dim=0)
            label_org = torch.cat((label_org_A, label_org_B), dim=0)
            seq_label = torch.cat((seq_label_A, seq_label_B), dim=0)


            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]


            c_org = label_org.clone()
            c_trg = label_trg.clone()  

            x_real = x_real.to(self.device)           # Input images.
            c_org = c_org.to(self.device)             # Original domain labels.
            c_trg = c_trg.to(self.device)             # Target domain labels.
            label_org = label_org.to(self.device)     # Labels for computing classification loss.
            label_trg = label_trg.to(self.device)     # Labels for computing classification loss.

            ##################### Setup Sequential Variables & Load###############################################3
            seq_label = self.label2onehot(seq_label,dim =2)
            seq_label = seq_label.to(self.device)     # Labels for computing Sequential classification loss.
            

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            out_src, out_cls, out_cls_TD = self.D(x_real)
            d_loss_real = - torch.mean(out_src)
            # d_loss_cls = self.classification_loss(out_cls, label_org, self.dataset)
            d_loss_cls = self.classification_loss(out_cls, label_org)

            d_td_loss = self.classification_loss(out_cls_TD, seq_label)

            # Compute loss with fake images.
            delta = self.G(x_real, c_trg)
            x_fake = torch.tanh(x_real + delta)
            out_src, out_cls, _ = self.D(x_fake.detach())
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            d_loss = d_td_loss + d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_gp'] = d_loss_gp.item()

            loss['D/td_loss'] = d_td_loss.item()

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #
            

            if (i+1) % self.n_critic == 0:
                # Original-to-target domain.
                delta = self.G(x_real, c_trg)
                x_fake = torch.tanh(x_real + delta)
                out_src, out_cls, out_cls_TD = self.D(x_fake)
                g_loss_fake = - torch.mean(out_src)
                g_loss_cls = self.classification_loss(out_cls, label_trg)

                g_loss_td_cls = self.classification_loss(out_cls_TD, seq_label)


                # Original-to-original domain.
                delta_id = self.G(x_real, c_org)
                x_fake_id = torch.tanh(x_real + delta_id)
                out_src_id, out_cls_id, _ = self.D(x_fake_id)
                g_loss_fake_id = - torch.mean(out_src_id)
                g_loss_cls_id = self.classification_loss(out_cls_id, label_org)
                g_loss_id = torch.mean(torch.abs(x_real - torch.tanh(delta_id + x_real)))


                # Target-to-original domain.
                delta_reconst = self.G(x_fake, c_org)
                x_reconst = torch.tanh(x_fake + delta_reconst)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))
         

                # Original-to-original domain.
                # delta_reconst_id = self.G(x_fake_id, c_org)
                # x_reconst_id = torch.tanh(x_fake_id + delta_reconst_id)
                # g_loss_rec_id = torch.mean(torch.abs(x_real - x_reconst_id))



                # Backward and optimize.

                # g_loss_same = g_loss_fake_id + self.lambda_rec * g_loss_rec_id + self.lambda_cls * g_loss_cls_id + self.lambda_id * g_loss_id
                g_loss_same = g_loss_fake_id  + self.lambda_cls * g_loss_cls_id + self.lambda_id * g_loss_id
                g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls + g_loss_same + self.lambda_TD * g_loss_td_cls

                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls'] = g_loss_cls.item()
                loss['G/loss_fake_id'] = g_loss_fake_id.item()
                # loss['G/loss_rec_id'] = g_loss_rec_id.item()
                loss['G/loss_cls_id'] = g_loss_cls_id.item()
                loss['G/loss_id'] = g_loss_id.item()

                loss['G/loss_td_cls'] = g_loss_td_cls.item()



            # =================================================================================== #
            #                                 5. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed]
                    for c_fixed in c_fixed_list:
                        delta = self.G(x_fixed, c_fixed)
                        x_fake_list.append(torch.tanh(delta + x_fixed))
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))


                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))


    def test(self):
        """Translate images using Fixed-Point GAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        data_loader = self.data_loader_B if self.direction == "B2A" else self.data_loader_A

        
        with torch.no_grad():
            for i, (x_real, label_org, seq_label,file_names) in enumerate(data_loader):

                x_real = x_real.to(self.device)
                c_trg = label_org.clone()
                c_trg[:, 0] = 0 if self.direction == "B2A" else 1       
                c_trg_list = [c_trg.to(self.device)]

                # Translate images.
                x_fake_list = []
                for c_trg in c_trg_list:

                    delta = self.G(x_real, c_trg)
                    x_fake_list.append( torch.tanh(delta + x_real) ) # generated image
                    


                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)

                for i in range(x_concat.size(0)):
                    seq_image = x_concat[i]
                    for j in range(seq_image.size(0)):
                        image = seq_image[j]
                        image_name = file_names[j][i]
                        result_path = os.path.join(self.result_dir, '{}'.format(image_name.split('/')[-1]))
                        save_image(self.denorm(image.data.cpu()), result_path, nrow=1, padding=0)
                        print('Saved real and fake images into {}...'.format(result_path))