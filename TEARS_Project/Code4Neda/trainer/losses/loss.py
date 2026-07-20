
import torch
import torch.nn as nn
import torch.nn.functional as F
import ot 
import numpy as np 

class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(BinaryCrossEntropyLoss, self).__init__()

    def forward(self, embeddings, positives):
        scores = F.binary_cross_entropy_with_logits(embeddings, positives.float(), reduction='none')
        return scores.mean(dim= 1).mean()

class BinaryCrossEntropyLossSoftmax(nn.Module):
    def __init__(self):
        super(BinaryCrossEntropyLossSoftmax, self).__init__()

    def forward(self, recon_x, x, mu=None, logvar=None, anneal=1.0):


        if logvar is not None:
            BCE = -torch.mean(torch.mean(F.log_softmax(recon_x, 1) * x, -1))

            KLD = -0.5 * torch.mean(torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            
        else:
            BCE = -torch.mean(torch.mean(F.log_softmax(recon_x, 1) * x, -1))
            KLD = torch.zeros_like(BCE)


        return BCE + anneal * KLD 

def swish(x):
    return x.mul(torch.sigmoid(x))

def log_norm_pdf(x, mu, logvar):

    # print(f"{torch.log(2 * torch.pi) + (x - mu).pow(2)=}")

    return -0.5*(logvar + torch.log(2 * torch.tensor(torch.pi)) + (x - mu).pow(2) / logvar.exp())

    
class MacridTEARSLLoss(nn.Module):
    def __init__(self,args):
        super(MacridTEARSLLoss, self).__init__()
        self.args = args
    def forward(self,recon_x, x,z, mu=None, logvar=None, anneal=1.0,prior =None ,tears = False,logits_text = None,logits_rec = None,prior_mu=None,prior_logvar=None,epsilon = 1,regularization_type = 'OT',gamma = .005,train_items= None):
        
        kfac = self.args.kfac
        split = 400//kfac
        
        BCE_merged = -torch.mean(torch.mean(F.log_softmax(recon_x, 1) * x, -1))
        BCE_text = -torch.mean(torch.mean(F.log_softmax(logits_text, 1) * x, -1))
        BCE_rec = -torch.mean(torch.mean(F.log_softmax(logits_rec, 1) * x, -1))
        

        BCE = (1/3)*(BCE_merged * self.args.recon_tau + BCE_text*self.args.text_tau + BCE_rec*self.args.rec_tau)
        
        logvar = logvar.view(-1,kfac,split)

        KLD1 = -0.5 * torch.mean(torch.mean(1 + logvar  - logvar.exp(), dim=-1))

        prior_logvar = prior_logvar.view(-1,kfac,split)

        KLD2 = -0.5 * torch.mean(torch.mean(1 + prior_logvar -  prior_logvar.exp(), dim=-1))

            
            # KLD2 = -0.5 * torch.mean(torch.mean(1 + prior_logvar - prior_mu.pow(2) - prior_logvar.exp(), dim=1))
        #reshape logvar back 
        logvar = logvar.view(-1,400)
        prior_logvar = prior_logvar.view(-1,400)
        sigma2 = torch.exp(logvar)
        sigma2_prior = torch.exp(prior_logvar)
        sigma2_diag = torch.stack([torch.diag(v) for v in sigma2])
        sigma2_prior_diag = torch.stack([torch.diag(v) for v in sigma2_prior])
            
        
        wasserstein_loss = self.bures_wasserstein_distance_vectorized(mu, sigma2_diag, prior_mu, sigma2_prior_diag).mean()
        if regularization_type =='none':
            l = BCE + (anneal/2) * (KLD1 + KLD2) 
        if regularization_type =='OT':
            l = BCE + (anneal/2) * (KLD1 + KLD2) +  epsilon*wasserstein_loss

    
            return l , BCE ,wasserstein_loss,BCE_rec,BCE_text,BCE_merged 

        else:
            mll = (F.log_softmax(recon_x, dim=-1) * x).sum(dim=-1).mean()
            kld = (log_norm_pdf(z, mu, logvar) - prior(x, z)).sum(dim=-1).mul(anneal).mean()
            negative_elbo = -(mll - kld)
            return negative_elbo
    def bures_wasserstein_distance_vectorized(self,means1, covs1, means2, covs2):
        # Compute squared L2 norm of differences in means
        mean_diff = means1 - means2
        mean_dist_squared = torch.sum(mean_diff ** 2, dim=1)  # Sum over columns to get a vector of shape (b,)

        # Cholesky decomposition of covs1 in batch
        sqrt_covs1 = torch.linalg.cholesky(covs1)

        # Batch matrix product: sqrt_covs1 * covs2 * sqrt_covs1
        # First part of the product: intermediate = sqrt_covs1 * covs2
        intermediate = torch.bmm(sqrt_covs1, covs2)
        # Second part of the product: product_matrix = intermediate * sqrt_covs1.transpose(1, 2)
        product_matrix = torch.bmm(intermediate, sqrt_covs1.transpose(1, 2))

        # Cholesky decomposition of the product_matrix in batch
        sqrt_middle = torch.linalg.cholesky(product_matrix)

        # Trace of sqrt_middle
        trace_sqrt_middle = torch.diagonal(sqrt_middle, dim1=-2, dim2=-1).sum(-1)

        # Trace of cov1 + cov2
        trace_covs1 = torch.diagonal(covs1, dim1=-2, dim2=-1).sum(-1)
        trace_covs2 = torch.diagonal(covs2, dim1=-2, dim2=-1).sum(-1)
        trace_cov_sum = trace_covs1 + trace_covs2

        # Total trace term
        trace_term = trace_cov_sum - 2 * trace_sqrt_middle

        # Total Bures-Wasserstein distance
        distance = mean_dist_squared + trace_term
        return distance


class MacridTEARSKL(nn.Module):
    def __init__(self, args):
        super(MacridTEARSKL, self).__init__()
        self.args = args

    def forward(self, recon_x, x, z, mu=None, logvar=None, anneal=1.0, prior=None, tears=False, 
                logits_text=None, logits_rec=None, prior_mu=None, prior_logvar=None, epsilon=1, 
                regularization_type='KL', gamma=0.005, train_items=None):
        
        kfac = self.args.kfac
        split = 400 // kfac
        
        # Reconstruction loss
        BCE_merged = -torch.mean(torch.mean(F.log_softmax(recon_x, 1) * x, -1))
        BCE_text = -torch.mean(torch.mean(F.log_softmax(logits_text, 1) * x, -1))
        BCE_rec = -torch.mean(torch.mean(F.log_softmax(logits_rec, 1) * x, -1))
        BCE = (1 / 3) * (BCE_merged * self.args.recon_tau + BCE_text * self.args.text_tau + BCE_rec * self.args.rec_tau)
        
        # Reshape logvar for covariance matrices
        logvar = logvar.view(-1, kfac, split)
        prior_logvar = prior_logvar.view(-1, kfac, split)
        cov1 = torch.diag_embed(torch.exp(logvar))  # Covariance matrices for the first Gaussian
        cov2 = torch.diag_embed(torch.exp(prior_logvar))  # Covariance matrices for the prior Gaussian

        # Analytical KL divergence between multivariate Gaussians
        diff = (mu - prior_mu).unsqueeze(-1)  # (batch, dim, 1)
        cov2_inv = torch.linalg.inv(cov2)  # Inverse of prior covariance matrix
        
        # KL divergence calculation
        trace_term = torch.einsum('bii->b', torch.matmul(cov2_inv, cov1))  # Trace of Σ2^-1 Σ1
        mean_term = torch.einsum('bij,bjk,bik->b', diff.transpose(-1, -2), cov2_inv, diff)  # Mahalanobis distance
        log_det_cov1 = torch.logdet(cov1)
        log_det_cov2 = torch.logdet(cov2)
        dim = mu.size(1)  # Dimensionality
        log_det_term = log_det_cov2 - log_det_cov1  # Log determinant term

        kl_divergence = 0.5 * (trace_term + mean_term - dim + log_det_term).mean()

        # Total loss
        if regularization_type == 'none':
            l = BCE + (anneal / 2) * kl_divergence
        elif regularization_type == 'KL':
            l = BCE + (anneal / 2) * kl_divergence + epsilon * kl_divergence

        return l, BCE, kl_divergence, BCE_rec, BCE_text, BCE_merged


class MacridTEARSJS_loss(nn.Module):
    def __init__(self, args):
        super(MacridTEARSJS_loss, self).__init__()
        self.args = args
    
    def forward(self, recon_x, x, z, mu=None, logvar=None, anneal=1.0, prior=None, tears=False, 
                logits_text=None, logits_rec=None, prior_mu=None, prior_logvar=None, 
                epsilon=1, regularization_type='JS', gamma=0.005, train_items=None):
        
        kfac = self.args.kfac
        split = 400 // kfac
        
        # BCE calculations (same as before)
        BCE_merged = -torch.mean(torch.mean(F.log_softmax(recon_x, 1) * x, -1))
        BCE_text = -torch.mean(torch.mean(F.log_softmax(logits_text, 1) * x, -1))
        BCE_rec = -torch.mean(torch.mean(F.log_softmax(logits_rec, 1) * x, -1))
        
        BCE = (1/3) * (BCE_merged * self.args.recon_tau + 
                       BCE_text * self.args.text_tau + 
                       BCE_rec * self.args.rec_tau)
        
        # Reshape logvars
        logvar = logvar.view(-1, kfac, split)
        prior_logvar = prior_logvar.view(-1, kfac, split)
        
        # KL Divergence calculations
        KLD1 = -0.5 * torch.mean(torch.mean(1 + logvar - logvar.exp(), dim=-1))
        KLD2 = -0.5 * torch.mean(torch.mean(1 + prior_logvar - prior_logvar.exp(), dim=-1))
        
        # Reshape back
        logvar = logvar.view(-1, 400)
        prior_logvar = prior_logvar.view(-1, 400)
        
        # Compute covariance matrices
        sigma2 = torch.exp(logvar)
        sigma2_prior = torch.exp(prior_logvar)
        
        # Create block diagonal covariance matrices
        sigma2_diag = torch.stack([torch.diag(v) for v in sigma2])
        sigma2_prior_diag = torch.stack([torch.diag(v) for v in sigma2_prior])
        
        # JS Divergence calculation
        js_divergence = self.js_divergence_multivariate(mu, sigma2, prior_mu, sigma2_prior).mean()
        
        # Loss calculation with different regularization types
        if regularization_type == 'none':
            l = BCE + (anneal/2) * (KLD1 + KLD2)
        elif regularization_type == 'JS':
            l = BCE + (anneal/2) * (KLD1 + KLD2) + epsilon * js_divergence
        else:
            mll = (F.log_softmax(recon_x, dim=-1) * x).sum(dim=-1).mean()
            kld = (log_norm_pdf(z, mu, logvar) - prior(x, z)).sum(dim=-1).mul(anneal).mean()
            l = -(mll - kld)
        

        return l, BCE, js_divergence, BCE_rec, BCE_text, BCE_merged
        
    def js_divergence_multivariate(self, mu1, cov1, mu2, cov2):
        """
        Compute Jensen-Shannon divergence between two multivariate Gaussian distributions
        
        Parameters:
        - mu1, mu2: Mean vectors (shape: [batch_size, d])
        - cov1, cov2: Covariance matrices (shape: [batch_size, d, d])
        
        Returns:
        - JS divergence (shape: [batch_size])
        """
        # Dimension of the distribution
        d = mu1.shape[-1]
        
        # Compute the mixed distribution
        mu_m = 0.5 * (mu1 + mu2)
        cov_m = 0.5 * (cov1 + cov2)
     
        def kl_divergence(mu_p, cov_p, mu_q, cov_q):
            """
            Compute KL divergence between two multivariate Gaussians
            with diagonal covariance matrices
            """
            # Variance ratio
            var_ratio = cov_p / cov_q
            
            # Mean difference
            mean_diff = ((mu_p - mu_q) ** 2) / cov_q
            
            # Logarithmic term
            log_term = torch.log(cov_q) - torch.log(cov_p)
            
            # KL divergence calculation
            kl_div = 0.5 * torch.sum(var_ratio + mean_diff - 1 + log_term, dim=1)
            
            return kl_div
        kl_p_m = kl_divergence(mu1, cov1, mu_m, cov_m)
        kl_q_m = kl_divergence(mu2, cov2, mu_m, cov_m)


        
        js_div = 0.5 * (kl_p_m + kl_q_m)
        
        return js_div
    
    

class MacridTEARSCL_loss(nn.Module):
    def __init__(self, args):
        super(MacridTEARSCL_loss, self).__init__()
        self.args = args
        self.similarity_fn = nn.CosineSimilarity(dim=-1)
    
    def forward(self, recon_x, x, z, mu=None, logvar=None, anneal=1.0, prior=None, tears=False, 
                logits_text=None, logits_rec=None, prior_mu=None, prior_logvar=None, 
                epsilon=1, regularization_type='JS', gamma=0.005, train_items=None):
        
        # BCE calculations (same as before)
        BCE_merged = -torch.mean(torch.mean(F.log_softmax(recon_x, 1) * x, -1))
        BCE_text = -torch.mean(torch.mean(F.log_softmax(logits_text, 1) * x, -1))
        BCE_rec = -torch.mean(torch.mean(F.log_softmax(logits_rec, 1) * x, -1))
        
        BCE =  (BCE_merged * self.args.recon_tau + 
                       BCE_text * self.args.text_tau + 
                       BCE_rec * self.args.rec_tau)
        
        KLD1 = -0.5 * torch.mean(torch.mean(1 + logvar - logvar.exp(), dim=-1))
        KLD2 = -0.5 * torch.mean(torch.mean(1 + prior_logvar - prior_logvar.exp(), dim=-1))
        
        # Concatenate sigmas and mus columnwise
        
        
        
        
        
        prior_sigma = torch.exp(prior_logvar)
        sigma = torch.exp(logvar)
        prior_embeddings = torch.cat((prior_mu, prior_sigma), dim=1)
        embeddings = torch.cat((mu, sigma), dim=1)
        
        
        
        # Compute similarity matrix
        batch_size = mu.shape[0]
        embeddings_expanded = embeddings.unsqueeze(1).expand(batch_size, batch_size, -1)
        prior_embeddings_transposed = prior_embeddings.unsqueeze(0).expand(batch_size, batch_size, -1)
        sim_matrix = self.similarity_fn(embeddings_expanded, prior_embeddings_transposed)
        
        # Compute contrastive loss
       # Compute contrastive loss
        # mask = torch.eye(batch_size, device=sim_matrix.device).bool()  # mask out diagonal elements
        exp_sim = torch.exp(sim_matrix) 

        # here we vary by summaries but one could vary by the  RS embeddings but we only backprop through the summaries so this makes more sense 
        #For the other implemmentation switch dim=1
        log_prob = sim_matrix - torch.log(torch.sum(exp_sim, dim=0)) 
        contrastive_loss = -torch.mean(log_prob.diag())
                
        # Loss calculation
        l = 1/3*BCE + 1/3*epsilon*contrastive_loss + 1/3*(KLD1 + KLD2)
        
        return l, BCE, contrastive_loss, BCE_rec, BCE_text, BCE_merged
    
    
    
class RecVAE_loss(nn.Module):
    def __init__(self,args):
        super(RecVAE_loss, self).__init__()
        self.args = args
    def forward(self,recon_x, x,z, mu=None, logvar=None, anneal=1.0,prior =None ,tears = False,logits_text = None,logits_rec = None,prior_mu=None,prior_logvar=None,epsilon = 1,regularization_type = 'OT',gamma = .005,train_items= None):
            norm = train_items.sum(dim=-1)
            kl_weight = gamma * norm

            
            if tears:
                if not self.args.no_merged:
                    BCE_merged = -torch.mean(torch.mean(F.log_softmax(recon_x, 1) * x, -1)) 
                else:
                    BCE_merged = torch.tensor(0).to(logits_rec.device)
                if not self.args.no_text:
                    BCE_text = -torch.mean(torch.mean(F.log_softmax(logits_text, 1) * x, -1))
                else:
                    BCE_text = torch.tensor(0).to(logits_rec.device)
                if not self.args.no_rec:
                    BCE_rec = -torch.mean(torch.mean(F.log_softmax(logits_rec, 1) * x, -1))
                else: 
                    BCE_rec = torch.tensor(0).to(logits_rec.device)

                
                BCE = (1/3)*(BCE_merged * self.args.recon_tau + BCE_text*self.args.text_tau + BCE_rec*self.args.rec_tau)
                
                if not self.args.KLD:
                    KLD1 = -0.5 * torch.mean(torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
                else: 
                    KLD1 = torch.tensor(0).to(logits_rec.device)

               
                KLD2 = (log_norm_pdf(z, prior_mu, prior_logvar) - prior( x,z)).mean(dim=-1).mul(anneal).mean()
                
                # KLD2 = -0.5 * torch.mean(torch.mean(1 + prior_logvar - prior_mu.pow(2) - prior_logvar.exp(), dim=1))

                sigma2 = torch.exp(logvar)

                sigma2_prior = torch.exp(prior_logvar)
                sigma2_diag = torch.stack([torch.diag(v) for v in sigma2])
                sigma2_prior_diag = torch.stack([torch.diag(v) for v in sigma2_prior])
                
            
                wasserstein_loss = self.bures_wasserstein_distance_vectorized(mu, sigma2_diag, prior_mu, sigma2_prior_diag).mean()
                if regularization_type =='none':
                    l = BCE + (anneal/2) * (KLD1 + KLD2) 
                if regularization_type =='OT':
                    l = BCE + (anneal/2) * (KLD1 + KLD2) +  epsilon*wasserstein_loss
        
            
                return l , BCE ,wasserstein_loss,BCE_rec,BCE_text,BCE_merged 

            else:
                mll = (F.log_softmax(recon_x, dim=-1) * x).sum(dim=-1).mean()
                kld = (log_norm_pdf(z, mu, logvar) - prior(x, z)).sum(dim=-1).mul(kl_weight).mean()
                negative_elbo = -(mll - kld)
                return negative_elbo
    def bures_wasserstein_distance_vectorized(self,means1, covs1, means2, covs2):
        # Compute squared L2 norm of differences in means
        mean_diff = means1 - means2
        mean_dist_squared = torch.sum(mean_diff ** 2, dim=1)  # Sum over columns to get a vector of shape (b,)

        # Cholesky decomposition of covs1 in batch
        #add small value to avoid singular matrix
        # covs1 = covs1 
        # print(f"{covs1.shape=}")
        # print(f"{covs1=}")
        # exit()
  
        sqrt_covs1 = torch.linalg.cholesky(covs1)

        # Batch matrix product: sqrt_covs1 * covs2 * sqrt_covs1
        # First part of the product: intermediate = sqrt_covs1 * covs2
        intermediate = torch.bmm(sqrt_covs1, covs2)
        # Second part of the product: product_matrix = intermediate * sqrt_covs1.transpose(1, 2)
        product_matrix = torch.bmm(intermediate, sqrt_covs1.transpose(1, 2))

        # Cholesky decomposition of the product_matrix in batch
        sqrt_middle = torch.linalg.cholesky(product_matrix)

        # Trace of sqrt_middle
        trace_sqrt_middle = torch.diagonal(sqrt_middle, dim1=-2, dim2=-1).sum(-1)

        # Trace of cov1 + cov2
        trace_covs1 = torch.diagonal(covs1, dim1=-2, dim2=-1).sum(-1)
        trace_covs2 = torch.diagonal(covs2, dim1=-2, dim2=-1).sum(-1)
        trace_cov_sum = trace_covs1 + trace_covs2

        # Total trace term
        trace_term = trace_cov_sum - 2 * trace_sqrt_middle

        # Total Bures-Wasserstein distance
        distance = mean_dist_squared + trace_term
        return distance


class PriorBCE(nn.Module):
    def __init__(self):
        super(PriorBCE, self).__init__()

    def forward(self, recon_x, logits_rec,logits_text,x, mu=None, logvar=None,prior_mu=None,prior_logvar=None,P=None,S =None,anneal=1.0,epsilon = 1,regularization_type = 'none'):


       

        BCE_merged = -torch.mean(torch.mean(F.log_softmax(recon_x, 1) * x, -1))
        BCE_text = -torch.mean(torch.mean(F.log_softmax(logits_text, 1) * x, -1))
        BCE_rec = -torch.mean(torch.mean(F.log_softmax(logits_rec, 1) * x, -1))

        BCE = (1/3)*(BCE_merged + BCE_text + BCE_rec)

        KLD1 = -0.5 * torch.mean(torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        KLD2 = -0.5 * torch.mean(torch.mean(1 + prior_logvar - prior_mu.pow(2) - prior_logvar.exp(), dim=1))

        sigma2 = torch.exp(logvar)
        sigma2_prior = torch.exp(prior_logvar)
        sigma2_diag = torch.stack([torch.diag(v) for v in sigma2])
        sigma2_prior_diag = torch.stack([torch.diag(v) for v in sigma2_prior])
        
       
        wasserstein_loss = self.bures_wasserstein_distance_vectorized(mu, sigma2_diag, prior_mu, sigma2_prior_diag).mean()
        if regularization_type =='none':
            l = BCE + (anneal/2) * (KLD1 + KLD2) 
        if regularization_type =='OT':
            l = BCE + (anneal/2) * (KLD1 + KLD2) +  epsilon*wasserstein_loss
      
            


        
        return l , BCE ,wasserstein_loss,BCE_rec,BCE_text,BCE_merged
    def bures_wasserstein_distance_vectorized(self,means1, covs1, means2, covs2):
        # Compute squared L2 norm of differences in means
        mean_diff = means1 - means2
        mean_dist_squared = torch.sum(mean_diff ** 2, dim=1)  # Sum over columns to get a vector of shape (b,)

        # Cholesky decomposition of covs1 in batch
        sqrt_covs1 = torch.linalg.cholesky(covs1)

        # Batch matrix product: sqrt_covs1 * covs2 * sqrt_covs1
        # First part of the product: intermediate = sqrt_covs1 * covs2
        intermediate = torch.bmm(sqrt_covs1, covs2)
        # Second part of the product: product_matrix = intermediate * sqrt_covs1.transpose(1, 2)
        product_matrix = torch.bmm(intermediate, sqrt_covs1.transpose(1, 2))

        # Cholesky decomposition of the product_matrix in batch
        sqrt_middle = torch.linalg.cholesky(product_matrix)

        # Trace of sqrt_middle
        trace_sqrt_middle = torch.diagonal(sqrt_middle, dim1=-2, dim2=-1).sum(-1)

        # Trace of cov1 + cov2
        trace_covs1 = torch.diagonal(covs1, dim1=-2, dim2=-1).sum(-1)
        trace_covs2 = torch.diagonal(covs2, dim1=-2, dim2=-1).sum(-1)
        trace_cov_sum = trace_covs1 + trace_covs2

        # Total trace term
        trace_term = trace_cov_sum - 2 * trace_sqrt_middle

        # Total Bures-Wasserstein distance
        distance = mean_dist_squared + trace_term
        return distance

    
    
    def bures_wasserstein_distance_vectorized(self,means1, covs1, means2, covs2):
        # Compute squared L2 norm of differences in means
        mean_diff = means1 - means2
        mean_dist_squared = torch.sum(mean_diff ** 2, dim=1)  # Sum over columns to get a vector of shape (b,)

        # Cholesky decomposition of covs1 in batch
        sqrt_covs1 = torch.linalg.cholesky(covs1)

        # Batch matrix product: sqrt_covs1 * covs2 * sqrt_covs1
        # First part of the product: intermediate = sqrt_covs1 * covs2
        intermediate = torch.bmm(sqrt_covs1, covs2)
        # Second part of the product: product_matrix = intermediate * sqrt_covs1.transpose(1, 2)
        product_matrix = torch.bmm(intermediate, sqrt_covs1.transpose(1, 2))

        # Cholesky decomposition of the product_matrix in batch
        sqrt_middle = torch.linalg.cholesky(product_matrix)

        # Trace of sqrt_middle
        trace_sqrt_middle = torch.diagonal(sqrt_middle, dim1=-2, dim2=-1).sum(-1)

        # Trace of cov1 + cov2
        trace_covs1 = torch.diagonal(covs1, dim1=-2, dim2=-1).sum(-1)
        trace_covs2 = torch.diagonal(covs2, dim1=-2, dim2=-1).sum(-1)
        trace_cov_sum = trace_covs1 + trace_covs2

        # Total trace term
        trace_term = trace_cov_sum - 2 * trace_sqrt_middle

        # Total Bures-Wasserstein distance
        distance = mean_dist_squared + trace_term
        return distance

class KLDivergenceLoss(nn.Module):
    def __init__(self,args,beta = .5):
        super(KLDivergenceLoss,self ).__init__()
        self.temp = args.temp
        #setup anneal schedule 
        self.BCE = BinaryCrossEntropyLossSoftmax()
        self.beta = beta
        self.anneal = args.anneal
        self.min_beta = .0


    def forward(self, embeddings, positives,beta = .5):
        

        probs = F.log_softmax(embeddings/self.temp, dim = 1)
        targets = F.softmax(positives.float()/self.temp, dim = 1)

        scores = F.kl_div(probs, targets, reduction='none')
        bce_loss = self.BCE(embeddings, positives)
        
        total_loss = self.beta * scores.sum(dim=1).mean() + (1-self.beta)* bce_loss
        return total_loss, scores.mean(dim= 1).mean(),bce_loss
    def anneal_beta(self,step, max_step):
        if self.anneal:
            return min(self.min_beta, self.beta * (step/max_step))
        else:
            return self.beta
    

def get_loss(loss_name, args=None):

    if loss_name == 'bce':
        return BinaryCrossEntropyLoss()
    if loss_name =='contrastive':


        return MacridTEARSCL_loss(args)

    elif loss_name == 'bce_softmax':
        return BinaryCrossEntropyLossSoftmax()
    elif loss_name == 'kl':
        return KLDivergenceLoss(args)
    elif loss_name == 'prior_bce':
        return PriorBCE()
    elif loss_name == 'RecVAE_loss':
        return RecVAE_loss(args)
    elif loss_name == 'Macrid_loss':
        return MacridTEARSLLoss(args)
    elif loss_name == 'MacridTEARSKL_loss':
        return MacridTEARSKL(args)
    elif loss_name == 'MacridTEARSJS_loss':

        return MacridTEARSJS_loss(args)
    else:
        raise ValueError(f"loss {loss_name} not supported")

