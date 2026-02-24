import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import initializers

import numpy as np
from pyDOE import lhs
import time

tf.random.set_seed(1234)

def get_ibc_and_inner_data(num_samples):
    
    z_data = tf.linspace(-1.0,1.0,101)
    y_data = tf.linspace(0.0,2.*np.pi,51)

    Y, Z = np.meshgrid(y_data,z_data)
    grid_loc = np.hstack((Y.flatten()[:,None], Z.flatten()[:,None]))

    lb = grid_loc.min(0)
    ub = grid_loc.max(0)
    
    grid_bottom = np.hstack((Y[0:1,:].T,  Z[0:1,:].T))
    grid_top    = np.hstack((Y[-1:,:].T,  Z[-1:,:].T))
    grid_left   = np.hstack((Y[1:-1,0:1], Z[1:-1,0:1]))
    grid_right  = np.hstack((Y[1:-1,-1:], Z[1:-1,-1:]))

    yb_top    = grid_top[:, 0:1]
    zb_top    = grid_top[:, 1:2]

    yb_bottom = grid_bottom[:, 0:1]
    zb_bottom = grid_bottom[:, 1:2]

    yb_left   = grid_left[:, 0:1]
    zb_left   = grid_left[:, 1:2]

    yb_right  = grid_right[:, 0:1]
    zb_right  = grid_right[:, 1:2]

    #Boundary conditions
    ur_bottom  = np.zeros((np.shape(grid_bottom)[0],1))
    vr_bottom  = np.zeros((np.shape(grid_bottom)[0],1))
    wr_bottom  = np.zeros((np.shape(grid_bottom)[0],1))
    ui_bottom  = np.zeros((np.shape(grid_bottom)[0],1))
    vi_bottom  = np.zeros((np.shape(grid_bottom)[0],1))
    wi_bottom  = np.zeros((np.shape(grid_bottom)[0],1))
    fxr_bottom = np.zeros((np.shape(grid_bottom)[0],1))
    fyr_bottom = np.zeros((np.shape(grid_bottom)[0],1))
    fzr_bottom = np.zeros((np.shape(grid_bottom)[0],1))
    fxi_bottom = np.zeros((np.shape(grid_bottom)[0],1))
    fyi_bottom = np.zeros((np.shape(grid_bottom)[0],1))
    fzi_bottom = np.zeros((np.shape(grid_bottom)[0],1))

    ur_top  = np.zeros((np.shape(grid_top)[0],1))
    vr_top  = np.zeros((np.shape(grid_top)[0],1))
    wr_top  = np.zeros((np.shape(grid_top)[0],1))
    ui_top  = np.zeros((np.shape(grid_top)[0],1))
    vi_top  = np.zeros((np.shape(grid_top)[0],1))
    wi_top  = np.zeros((np.shape(grid_top)[0],1))
    fxr_top = np.zeros((np.shape(grid_top)[0],1))
    fyr_top = np.zeros((np.shape(grid_top)[0],1))
    fzr_top = np.zeros((np.shape(grid_top)[0],1))
    fxi_top = np.zeros((np.shape(grid_top)[0],1))
    fyi_top = np.zeros((np.shape(grid_top)[0],1))
    fzi_top = np.zeros((np.shape(grid_top)[0],1))

    grid_f_train = lb + (ub-lb)*lhs(2, num_samples)
    grid_train = np.vstack((grid_bottom, grid_top, grid_left, grid_right, grid_f_train))

    ur_ob = np.vstack([ur_bottom, ur_top])
    vr_ob = np.vstack([vr_bottom, vr_top])
    wr_ob = np.vstack([wr_bottom, wr_top])
    ui_ob = np.vstack([ui_bottom, ui_top])
    vi_ob = np.vstack([vi_bottom, vi_top])
    wi_ob = np.vstack([wi_bottom, wi_top])
    fxr_ob = np.vstack([fxr_bottom, fxr_top])
    fyr_ob = np.vstack([fyr_bottom, fyr_top])
    fzr_ob = np.vstack([fzr_bottom, fzr_top])
    fxi_ob = np.vstack([fxi_bottom, fxi_top])
    fyi_ob = np.vstack([fyi_bottom, fyi_top])
    fzi_ob = np.vstack([fzi_bottom, fzi_top])

    yd = grid_train[:, 0:1]
    zd = grid_train[:, 1:2]

    return  yb_top, zb_top, yb_bottom, zb_bottom, yb_left, zb_left, yb_right, zb_right, yd, zd, \
            ur_ob, vr_ob, wr_ob, ui_ob, vi_ob, wi_ob, \
            fxr_ob, fyr_ob, fzr_ob, fxi_ob, fyi_ob, fzi_ob

class PdeModel:
    def __init__(self, inputs, outputs, get_models, loss_fn,
                 optimizer, metrics, parameters, hyperparam_filename, hyperparam_expname, ibc_layer=False, mask=None):
        self.nn_model = get_models['nn_model']
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.ibc_layer = ibc_layer

        self.hyperparam_filename = hyperparam_filename
        self.hyperparam_expname  = hyperparam_expname

        self.yin       = tf.constant(inputs['yin'],       dtype=tf.float32)
        self.zin       = tf.constant(inputs['zin'],       dtype=tf.float32)
        self.yb_top    = tf.constant(inputs['yb_top'],    dtype=tf.float32)
        self.zb_top    = tf.constant(inputs['zb_top'],    dtype=tf.float32)
        self.yb_bottom = tf.constant(inputs['yb_bottom'], dtype=tf.float32)
        self.zb_bottom = tf.constant(inputs['zb_bottom'], dtype=tf.float32)
        self.yb_left   = tf.constant(inputs['yb_left'],   dtype=tf.float32)
        self.zb_left   = tf.constant(inputs['zb_left'],   dtype=tf.float32)
        self.yb_right  = tf.constant(inputs['yb_right'],  dtype=tf.float32)
        self.zb_right  = tf.constant(inputs['zb_right'],  dtype=tf.float32)
        
        self.yb = tf.concat([self.yb_top, self.yb_bottom],0)
        self.zb = tf.concat([self.zb_top, self.zb_bottom],0)

        self.urb  = tf.constant(outputs['urb'],  dtype=tf.float32)
        self.vrb  = tf.constant(outputs['vrb'],  dtype=tf.float32)
        self.wrb  = tf.constant(outputs['wrb'],  dtype=tf.float32)
        self.uib  = tf.constant(outputs['uib'],  dtype=tf.float32)
        self.vib  = tf.constant(outputs['vib'],  dtype=tf.float32)
        self.wib  = tf.constant(outputs['wib'],  dtype=tf.float32)
        self.fxrb = tf.constant(outputs['fxrb'], dtype=tf.float32)
        self.fyrb = tf.constant(outputs['fyrb'], dtype=tf.float32)
        self.fzrb = tf.constant(outputs['fzrb'], dtype=tf.float32)
        self.fxib = tf.constant(outputs['fxib'], dtype=tf.float32)
        self.fyib = tf.constant(outputs['fyib'], dtype=tf.float32)
        self.fzib = tf.constant(outputs['fzib'], dtype=tf.float32)

        self.loss_tracker              = metrics['loss']
        self.residual_loss_tracker     = metrics['residual_loss']
        self.residual_loss_ur_tracker  = metrics['residual_loss_ur']
        self.residual_loss_ui_tracker  = metrics['residual_loss_ui']
        self.residual_loss_vr_tracker  = metrics['residual_loss_vr']
        self.residual_loss_vi_tracker  = metrics['residual_loss_vi']
        self.residual_loss_wr_tracker  = metrics['residual_loss_wr']
        self.residual_loss_wi_tracker  = metrics['residual_loss_wi']
        self.residual_loss_cr_tracker  = metrics['residual_loss_cr']
        self.residual_loss_ci_tracker  = metrics['residual_loss_ci']
        self.bc_loss_tracker           = metrics['bc_loss']
        self.G_loss_tracker            = metrics['G_loss']
        self.G_loss1_tracker           = metrics['G_loss1']
        self.G_loss2_tracker           = metrics['G_loss2']

        self.Re    = tf.constant(parameters['Re'], dtype=tf.float32)
        self.kx    = tf.constant(parameters['kx'], dtype=tf.float32)
        self.omega = tf.constant(parameters['omega'], dtype=tf.float32)
       
        self.U = lambda y, z: 1.0 - tf.square(z) - 0.*y

    @tf.function
    def pde_residual(self, training=True):

        with tf.GradientTape(persistent=True) as outer_tape:
            outer_tape.watch([self.yin,self.zin])
            with tf.GradientTape(persistent=True) as inner_tape:
                inner_tape.watch([self.yin,self.zin])
                ur, vr, wr, pr, ui, vi, wi, pi, fxr, fyr, fzr, fxi, fyi, fzi = \
                        self.nn_model([self.yin,self.zin], training=True)

                Re    = self.Re
                kx    = self.kx
                omega = self.omega
                U     = self.U(self.yin,self.zin)

            #First order derivatives
            ur_z = inner_tape.gradient(ur, self.zin)
            vr_z = inner_tape.gradient(vr, self.zin)
            wr_z = inner_tape.gradient(wr, self.zin)
            pr_z = inner_tape.gradient(pr, self.zin)
            ui_z = inner_tape.gradient(ui, self.zin)
            vi_z = inner_tape.gradient(vi, self.zin)
            wi_z = inner_tape.gradient(wi, self.zin)
            pi_z = inner_tape.gradient(pi, self.zin)

            ur_y = inner_tape.gradient(ur, self.yin)
            vr_y = inner_tape.gradient(vr, self.yin)
            wr_y = inner_tape.gradient(wr, self.yin)
            pr_y = inner_tape.gradient(pr, self.yin)
            ui_y = inner_tape.gradient(ui, self.yin)
            vi_y = inner_tape.gradient(vi, self.yin)
            wi_y = inner_tape.gradient(wi, self.yin)
            pi_y = inner_tape.gradient(pi, self.yin)

            dU_z = inner_tape.gradient(U, self.zin)
            dU_y = inner_tape.gradient(U, self.yin)

        ur_zz = outer_tape.gradient(ur_z, self.zin)
        vr_zz = outer_tape.gradient(vr_z, self.zin)
        wr_zz = outer_tape.gradient(wr_z, self.zin)
        ui_zz = outer_tape.gradient(ui_z, self.zin)
        vi_zz = outer_tape.gradient(vi_z, self.zin)
        wi_zz = outer_tape.gradient(wi_z, self.zin)
        pr_zz = outer_tape.gradient(pr_z, self.zin)
        pi_zz = outer_tape.gradient(pi_z, self.zin)

        ur_yy = outer_tape.gradient(ur_y, self.yin)
        vr_yy = outer_tape.gradient(vr_y, self.yin)
        wr_yy = outer_tape.gradient(wr_y, self.yin)
        ui_yy = outer_tape.gradient(ui_y, self.yin)
        vi_yy = outer_tape.gradient(vi_y, self.yin)
        wi_yy = outer_tape.gradient(wi_y, self.yin)
        pr_yy = outer_tape.gradient(pr_y, self.yin)
        pi_yy = outer_tape.gradient(pi_y, self.yin)

        Rur = - omega*ui - kx*U*ui + vr*dU_y + wr*dU_z - kx*pi - (1./Re)*ur_yy - (1./Re)*ur_zz + ((kx**2)/Re)*ur - fxr
        Rvr = - omega*vi - kx*U*vi                     + pr_y  - (1./Re)*vr_yy - (1./Re)*vr_zz + ((kx**2)/Re)*vr - fyr
        Rwr = - omega*wi - kx*U*wi                     + pr_z  - (1./Re)*wr_yy - (1./Re)*wr_zz + ((kx**2)/Re)*wr - fzr
        Rcr = - kx*ui + vr_y + wr_z

        Rui =   omega*ur + kx*U*ur + vi*dU_y + wi*dU_z + kx*pr - (1./Re)*ui_yy - (1./Re)*ui_zz + ((kx**2)/Re)*ui - fxi
        Rvi =   omega*vr + kx*U*vr                     + pi_y  - (1./Re)*vi_yy - (1./Re)*ui_zz + ((kx**2)/Re)*vi - fyi
        Rwi =   omega*wr + kx*U*wr                     + pi_z  - (1./Re)*wi_yy - (1./Re)*ui_zz + ((kx**2)/Re)*wi - fzi
        Rci =   kx*ur + vi_y + wi_z

        ur_sq = ur**2; int_ur = tf.reduce_sum( (ur_sq[:-1]+ur_sq[1:])/2. * tf.abs(self.zin[1:]-self.zin[:-1]) * tf.abs(self.yin[1:]-self.yin[:-1]) ) 
        vr_sq = vr**2; int_vr = tf.reduce_sum( (vr_sq[:-1]+vr_sq[1:])/2. * tf.abs(self.zin[1:]-self.zin[:-1]) * tf.abs(self.yin[1:]-self.yin[:-1]) )
        wr_sq = wr**2; int_wr = tf.reduce_sum( (wr_sq[:-1]+wr_sq[1:])/2. * tf.abs(self.zin[1:]-self.zin[:-1]) * tf.abs(self.yin[1:]-self.yin[:-1]) )
        ui_sq = ui**2; int_ui = tf.reduce_sum( (ui_sq[:-1]+ui_sq[1:])/2. * tf.abs(self.zin[1:]-self.zin[:-1]) * tf.abs(self.yin[1:]-self.yin[:-1]) )
        vi_sq = vi**2; int_vi = tf.reduce_sum( (vi_sq[:-1]+vi_sq[1:])/2. * tf.abs(self.zin[1:]-self.zin[:-1]) * tf.abs(self.yin[1:]-self.yin[:-1]) )
        wi_sq = wi**2; int_wi = tf.reduce_sum( (wi_sq[:-1]+wi_sq[1:])/2. * tf.abs(self.zin[1:]-self.zin[:-1]) * tf.abs(self.yin[1:]-self.yin[:-1]) )

        fxr_sq = fxr**2; int_fxr = tf.reduce_sum( (fxr_sq[:-1]+fxr_sq[1:])/2. * tf.abs(self.zin[1:]-self.zin[:-1]) * tf.abs(self.yin[1:]-self.yin[:-1]) )
        fyr_sq = fyr**2; int_fyr = tf.reduce_sum( (fyr_sq[:-1]+fyr_sq[1:])/2. * tf.abs(self.zin[1:]-self.zin[:-1]) * tf.abs(self.yin[1:]-self.yin[:-1]) )
        fzr_sq = fzr**2; int_fzr = tf.reduce_sum( (fzr_sq[:-1]+fzr_sq[1:])/2. * tf.abs(self.zin[1:]-self.zin[:-1]) * tf.abs(self.yin[1:]-self.yin[:-1]) )
        fxi_sq = fxi**2; int_fxi = tf.reduce_sum( (fxi_sq[:-1]+fxi_sq[1:])/2. * tf.abs(self.zin[1:]-self.zin[:-1]) * tf.abs(self.yin[1:]-self.yin[:-1]) )
        fyi_sq = fyi**2; int_fyi = tf.reduce_sum( (fyi_sq[:-1]+fyi_sq[1:])/2. * tf.abs(self.zin[1:]-self.zin[:-1]) * tf.abs(self.yin[1:]-self.yin[:-1]) )
        fzi_sq = fzi**2; int_fzi = tf.reduce_sum( (fzi_sq[:-1]+fzi_sq[1:])/2. * tf.abs(self.zin[1:]-self.zin[:-1]) * tf.abs(self.yin[1:]-self.yin[:-1]) )

        dur_sq = ur_y**2; int_dur = tf.reduce_sum( (dur_sq[:-1]+dur_sq[1:])/2. * tf.abs(self.zin[1:]-self.zin[:-1]) * tf.abs(self.yin[1:]-self.yin[:-1]) ) 
        dvr_sq = vr_y**2; int_dvr = tf.reduce_sum( (dvr_sq[:-1]+dvr_sq[1:])/2. * tf.abs(self.zin[1:]-self.zin[:-1]) * tf.abs(self.yin[1:]-self.yin[:-1]) )
        dwr_sq = wr_y**2; int_dwr = tf.reduce_sum( (dwr_sq[:-1]+dwr_sq[1:])/2. * tf.abs(self.zin[1:]-self.zin[:-1]) * tf.abs(self.yin[1:]-self.yin[:-1]) )
        dui_sq = ui_y**2; int_dui = tf.reduce_sum( (dui_sq[:-1]+dui_sq[1:])/2. * tf.abs(self.zin[1:]-self.zin[:-1]) * tf.abs(self.yin[1:]-self.yin[:-1]) )
        dvi_sq = vi_y**2; int_dvi = tf.reduce_sum( (dvi_sq[:-1]+dvi_sq[1:])/2. * tf.abs(self.zin[1:]-self.zin[:-1]) * tf.abs(self.yin[1:]-self.yin[:-1]) )
        dwi_sq = wi_y**2; int_dwi = tf.reduce_sum( (dwi_sq[:-1]+dwi_sq[1:])/2. * tf.abs(self.zin[1:]-self.zin[:-1]) * tf.abs(self.yin[1:]-self.yin[:-1]) )

        Gq  = (int_ur  + int_vr  + int_wr  + int_ui  + int_vi  + int_wi)
        Gf  = (int_fxr + int_fyr + int_fzr + int_fxi + int_fyi + int_fzi)
        Gdq = (int_dur + int_dvr + int_dwr + int_dui + int_dvi + int_dwi)

        #Sym_u  = tf.reduce_sum(((ur**2+ui**2) - (Sur**2+Sui**2))**2)
        #Sym_v  = tf.reduce_sum(((vr**2+vi**2) - (Svr**2+Svi**2))**2)
        #Sym_w  = tf.reduce_sum(((wr**2+wi**2) - (Swr**2+Swi**2))**2)
        #Sym_fx = tf.reduce_sum(((fxr**2+fxi**2) - (Sfxr**2+Sfxi**2))**2)
        #Sym_fy = tf.reduce_sum(((fyr**2+fyi**2) - (Sfyr**2+Sfyi**2))**2)
        #Sym_fz = tf.reduce_sum(((fzr**2+fzi**2) - (Sfzr**2+Sfzi**2))**2)

        #return Rur, Rvr, Rwr, Rcr, Rui, Rvi, Rwi, Rci, Gq, Gf, Sym_u, Sym_v, Sym_w, Sym_fx, Sym_fy, Sym_fz
        return Rur, Rvr, Rwr, Rcr, Rui, Rvi, Rwi, Rci, Gq, Gf, Gdq #, Sym_u, Sym_v, Sym_w, Sym_fx, Sym_fy, Sym_fz

    @tf.function
    def train_step(self):

        with tf.GradientTape(persistent=True) as tape:
            urb_pred, vrb_pred, wrb_pred, prb_pred, \
            uib_pred, vib_pred, wib_pred, pib_pred, \
            fxrb_pred, fyrb_pred, fzrb_pred, \
            fxib_pred, fyib_pred, fzib_pred = \
                self.nn_model([self.yb,self.zb], training=True)

            with tf.GradientTape(persistent=True) as innertape:
                innertape.watch([self.yb_left, self.yb_right])

                urb_pred_right, vrb_pred_right, wrb_pred_right, prb_pred_right, \
                uib_pred_right, vib_pred_right, wib_pred_right, pib_pred_right, \
                fxrb_pred_right, fyrb_pred_right, fzrb_pred_right, \
                fxib_pred_right, fyib_pred_right, fzib_pred_right = \
                    self.nn_model([self.yb_right,self.zb_right], training=True)

                urb_pred_left, vrb_pred_left, wrb_pred_left, prb_pred_left, \
                uib_pred_left, vib_pred_left, wib_pred_left, pib_pred_left, \
                fxrb_pred_left, fyrb_pred_left, fzrb_pred_left, \
                fxib_pred_left, fyib_pred_left, fzib_pred_left = \
                    self.nn_model([self.yb_left,self.zb_left], training=True)

            durb_pred_right = innertape.gradient(urb_pred_right, self.yb_right)
            dvrb_pred_right = innertape.gradient(vrb_pred_right, self.yb_right)
            dwrb_pred_right = innertape.gradient(wrb_pred_right, self.yb_right)
            duib_pred_right = innertape.gradient(uib_pred_right, self.yb_right)
            dvib_pred_right = innertape.gradient(vib_pred_right, self.yb_right)
            dwib_pred_right = innertape.gradient(wib_pred_right, self.yb_right)

            dfxrb_pred_right = innertape.gradient(fxrb_pred_right, self.yb_right)
            dfyrb_pred_right = innertape.gradient(fyrb_pred_right, self.yb_right)
            dfzrb_pred_right = innertape.gradient(fzrb_pred_right, self.yb_right)
            dfxib_pred_right = innertape.gradient(fxib_pred_right, self.yb_right)
            dfyib_pred_right = innertape.gradient(fyib_pred_right, self.yb_right)
            dfzib_pred_right = innertape.gradient(fzib_pred_right, self.yb_right)

            durb_pred_left  = innertape.gradient(urb_pred_left, self.yb_left)
            dvrb_pred_left  = innertape.gradient(vrb_pred_left, self.yb_left)
            dwrb_pred_left  = innertape.gradient(wrb_pred_left, self.yb_left)
            duib_pred_left  = innertape.gradient(uib_pred_left, self.yb_left)
            dvib_pred_left  = innertape.gradient(vib_pred_left, self.yb_left)
            dwib_pred_left  = innertape.gradient(wib_pred_left, self.yb_left)

            dfxrb_pred_left = innertape.gradient(fxrb_pred_left, self.yb_left)
            dfyrb_pred_left = innertape.gradient(fyrb_pred_left, self.yb_left)
            dfzrb_pred_left = innertape.gradient(fzrb_pred_left, self.yb_left)
            dfxib_pred_left = innertape.gradient(fxib_pred_left, self.yb_left)
            dfyib_pred_left = innertape.gradient(fyib_pred_left, self.yb_left)
            dfzib_pred_left = innertape.gradient(fzib_pred_left, self.yb_left)

            bc_loss1 = self.loss_fn(self.urb, urb_pred)   + self.loss_fn(self.uib, uib_pred) + \
                       self.loss_fn(self.vrb, vrb_pred)   + self.loss_fn(self.vib, vib_pred) + \
                       self.loss_fn(self.wrb, wrb_pred)   + self.loss_fn(self.wib, wib_pred) + \
                       self.loss_fn(self.fxrb, fxrb_pred) + self.loss_fn(self.fxib, fxib_pred) + \
                       self.loss_fn(self.fyrb, fyrb_pred) + self.loss_fn(self.fyib, fyib_pred) + \
                       self.loss_fn(self.fzrb, fzrb_pred) + self.loss_fn(self.fzib, fzib_pred) 

            bc_loss2 = self.loss_fn(urb_pred_right,  urb_pred_left)   + self.loss_fn(uib_pred_right, uib_pred_left) + \
                       self.loss_fn(vrb_pred_right,  vrb_pred_left)   + self.loss_fn(vib_pred_right, vib_pred_left) + \
                       self.loss_fn(wrb_pred_right,  wrb_pred_left)   + self.loss_fn(wib_pred_right, wib_pred_left) + \
                       self.loss_fn(fxrb_pred_right, fxrb_pred_left) + self.loss_fn(fxib_pred_right, fxib_pred_left) + \
                       self.loss_fn(fyrb_pred_right, fyrb_pred_left) + self.loss_fn(fyib_pred_right, fyib_pred_left) + \
                       self.loss_fn(fzrb_pred_right, fzrb_pred_left) + self.loss_fn(fzib_pred_right, fzib_pred_left) 

            bc_loss3 = self.loss_fn(durb_pred_right,  durb_pred_left)   + self.loss_fn(duib_pred_right, duib_pred_left) + \
                       self.loss_fn(dvrb_pred_right,  dvrb_pred_left)   + self.loss_fn(dvib_pred_right, dvib_pred_left) + \
                       self.loss_fn(dwrb_pred_right,  dwrb_pred_left)   + self.loss_fn(dwib_pred_right, dwib_pred_left) + \
                       self.loss_fn(dfxrb_pred_right, dfxrb_pred_left) + self.loss_fn(dfxib_pred_right, dfxib_pred_left) + \
                       self.loss_fn(dfyrb_pred_right, dfyrb_pred_left) + self.loss_fn(dfyib_pred_right, dfyib_pred_left) + \
                       self.loss_fn(dfzrb_pred_right, dfzrb_pred_left) + self.loss_fn(dfzib_pred_right, dfzib_pred_left) 

            bc_loss = bc_loss1 + bc_loss2 + bc_loss3

            Rur, Rvr, Rwr, Rcr, Rui, Rvi, Rwi, Rci, Gq, Gf, Gdq = self.pde_residual(True)

            f = open(self.hyperparam_filename)
            for line_num,line in enumerate(f):
                lsplit = line.split('\t')
                if line_num==0:
                    for title_ind,title in enumerate(lsplit):
                        if title.split('\n')[0] == '#name':
                            name_ind = title_ind
                        elif title.split('\n')[0] == 'beta1':
                            beta1_ind = title_ind
                        elif title.split('\n')[0] == 'beta2':
                            beta2_ind = title_ind
                        elif title.split('\n')[0] == 'alpha1':
                            alpha1_ind = title_ind
                        elif title.split('\n')[0] == 'alpha2':
                            alpha2_ind = title_ind
                        elif title.split('\n')[0] == 'alpha3':
                            alpha3_ind = title_ind
                        elif title.split('\n')[0] == 'alpha4':
                            alpha4_ind = title_ind
                        elif title.split('\n')[0] == 'Ra_ur':
                            Ralpha_ur_ind = title_ind
                        elif title.split('\n')[0] == 'Ra_ui':
                            Ralpha_ui_ind = title_ind
                        elif title.split('\n')[0] == 'Ra_vr':
                            Ralpha_vr_ind = title_ind
                        elif title.split('\n')[0] == 'Ra_vi':
                            Ralpha_vi_ind = title_ind
                        elif title.split('\n')[0] == 'Ra_wr':
                            Ralpha_wr_ind = title_ind
                        elif title.split('\n')[0] == 'Ra_wi':
                            Ralpha_wi_ind = title_ind
                        elif title.split('\n')[0] == 'Ra_cr':
                            Ralpha_cr_ind = title_ind
                        elif title.split('\n')[0] == 'Ra_ci':
                            Ralpha_ci_ind = title_ind
                else:      
                    if lsplit[name_ind] == self.hyperparam_expname:
                        num_list = [lsplit[ind].split('\n')[0] for ind in range(0,len(lsplit))]
                        beta1     = float(num_list[beta1_ind])
                        beta2     = float(num_list[beta2_ind])
                        alpha1    = float(num_list[alpha1_ind])
                        alpha2    = float(num_list[alpha2_ind])
                        alpha3    = float(num_list[alpha3_ind])
                        alpha4    = float(num_list[alpha4_ind])
                        Ralpha_ur = float(num_list[Ralpha_ur_ind])
                        Ralpha_ui = float(num_list[Ralpha_ui_ind])
                        Ralpha_vr = float(num_list[Ralpha_vr_ind])
                        Ralpha_vi = float(num_list[Ralpha_vi_ind])
                        Ralpha_wr = float(num_list[Ralpha_wr_ind])
                        Ralpha_wi = float(num_list[Ralpha_wi_ind])
                        Ralpha_cr = float(num_list[Ralpha_cr_ind])
                        Ralpha_ci = float(num_list[Ralpha_ci_ind])
                    else:
                        continue
            #Ralpha_ur = 10.;  Ralpha_ui = 10. 
            #Ralpha_cr = 100.;  Ralpha_ci = 100. 
            '''
            beta1 = 0.1
            beta2 = 1.
            Ralpha_ur = 1.
            Ralpha_ui = 1.
            Ralpha_vr = 1.
            Ralpha_vi = 1.
            Ralpha_wr = 1.
            Ralpha_wi = 1.
            Ralpha_cr = 1.
            Ralpha_ci = 1.
            alpha1 = 100.
            alpha2 = 1.
            alpha3 = 100.
            '''

            hyperparams = {'beta1':beta1, 'beta2':beta2, 'alpha1':alpha1, 'alpha2':alpha2, 'alpha3':alpha3, \
                    'Ralpha_ur':Ralpha_ur, 'Ralpha_ui':Ralpha_ui, 'Ralpha_vr':Ralpha_vr, 'Ralpha_vi':Ralpha_vi, \
                    'Ralpha_wr':Ralpha_wr, 'Ralpha_wi':Ralpha_wi, 'Ralpha_cr':Ralpha_cr, 'Ralpha_ci':Ralpha_ci}  
            np.savez('hyperparams.npz', **hyperparams)

            eps = 1e-8
            G_loss1 = - tf.math.log(Gq/(Gf+eps)+eps)
            #G_loss1 = - tf.math.log(Gq+eps)
            G_loss2 = (Gf-1.)**2
            G_loss = beta1 * G_loss1 + beta2 * G_loss2

            residual_loss_ur = tf.reduce_mean(tf.square(Rur)) 
            residual_loss_ui = tf.reduce_mean(tf.square(Rui))
            residual_loss_vr = tf.reduce_mean(tf.square(Rvr)) 
            residual_loss_vi = tf.reduce_mean(tf.square(Rvi))
            residual_loss_wr = tf.reduce_mean(tf.square(Rwr)) 
            residual_loss_wi = tf.reduce_mean(tf.square(Rwi))
            residual_loss_cr = tf.reduce_mean(tf.square(Rcr)) 
            residual_loss_ci = tf.reduce_mean(tf.square(Rci))
            residual_loss = Ralpha_ur * residual_loss_ur + Ralpha_ui * residual_loss_ui + \
                            Ralpha_vr * residual_loss_vr + Ralpha_vi * residual_loss_vi + \
                            Ralpha_wr * residual_loss_wr + Ralpha_wi * residual_loss_wi + \
                            Ralpha_cr * residual_loss_cr + Ralpha_ci * residual_loss_ci 

            
            loss = alpha1 * residual_loss + alpha2 * G_loss + alpha3 * bc_loss #+ 10.0/(Gdq+1e-6)

        grads = tape.gradient(loss, self.nn_model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.nn_model.trainable_weights))
        grad_bc  = tape.gradient(bc_loss, self.nn_model.trainable_weights)
        grad_res = tape.gradient(residual_loss, self.nn_model.trainable_weights)
        grad_G   = tape.gradient(G_loss, self.nn_model.trainable_weights)

        self.loss_tracker.update_state(loss)
        self.residual_loss_tracker.update_state(residual_loss)
        self.residual_loss_ur_tracker.update_state(residual_loss_ur)
        self.residual_loss_ui_tracker.update_state(residual_loss_ui)
        self.residual_loss_vr_tracker.update_state(residual_loss_vr)
        self.residual_loss_vi_tracker.update_state(residual_loss_vi)
        self.residual_loss_wr_tracker.update_state(residual_loss_wr)
        self.residual_loss_wi_tracker.update_state(residual_loss_wi)
        self.residual_loss_cr_tracker.update_state(residual_loss_cr)
        self.residual_loss_ci_tracker.update_state(residual_loss_ci)
        self.bc_loss_tracker.update_state(bc_loss)
        self.G_loss_tracker.update_state(G_loss)
        self.G_loss1_tracker.update_state(G_loss1)
        self.G_loss2_tracker.update_state(G_loss2)

        return {'loss':self.loss_tracker.result(), 'residual_loss':self.residual_loss_tracker.result(), \
                'G_loss':self.G_loss_tracker.result(), 'bc_loss':self.bc_loss_tracker.result(), \
                'residual_loss_ur':self.residual_loss_ur_tracker.result(), 'residual_loss_ui':self.residual_loss_ui_tracker.result(), \
                'residual_loss_vr':self.residual_loss_vr_tracker.result(), 'residual_loss_vi':self.residual_loss_vi_tracker.result(), \
                'residual_loss_wr':self.residual_loss_wr_tracker.result(), 'residual_loss_wi':self.residual_loss_wi_tracker.result(), \
                'residual_loss_cr':self.residual_loss_cr_tracker.result(), 'residual_loss_ci':self.residual_loss_ci_tracker.result(), \
                'G_loss1':self.G_loss1_tracker.result(), 'G_loss2':self.G_loss2_tracker.result()}, \
                grads, grad_bc, grad_res, grad_G

    def reset_metrics(self):        
        self.loss_tracker.reset_state()
        self.residual_loss_tracker.reset_state()
        self.residual_loss_ur_tracker.reset_state()
        self.residual_loss_ui_tracker.reset_state()
        self.residual_loss_vr_tracker.reset_state()
        self.residual_loss_vi_tracker.reset_state()
        self.residual_loss_wr_tracker.reset_state()
        self.residual_loss_wi_tracker.reset_state()
        self.residual_loss_cr_tracker.reset_state()
        self.residual_loss_ci_tracker.reset_state()
        self.bc_loss_tracker.reset_state()
        self.G_loss_tracker.reset_state()
        self.G_loss1_tracker.reset_state()
        self.G_loss2_tracker.reset_state()

    def run(self, epochs, proj_name, log_dir, wb=False, verbose_freq=1000, grad_freq=5000):

        self.reset_metrics()
        history = {"loss": [], "residual_loss": [], "bc_loss": [], "G_loss": [],  \
                "residual_loss_ur":[], "residual_loss_ui":[], \
                "residual_loss_vr":[], "residual_loss_vi":[], \
                "residual_loss_wr":[], "residual_loss_wi":[], \
                "residual_loss_cr":[], "residual_loss_ci":[], \
                "G_loss1": [], "G_loss2": []}

        start_time = time.time()

        for epoch in range(epochs):
            
            logs, grads, grad_bc, grad_res, grad_G = self.train_step()
            if (epoch+1) % grad_freq == 0:
                default_grad = tf.zeros([1])
                grad_bc  = [grad if grad is not None else default_grad for grad in grad_bc]
                grad_res = [grad if grad is not None else default_grad for grad in grad_res]
                grad_G   = [grad if grad is not None else default_grad for grad in grad_G]
                grads    = [grad if grad is not None else default_grad for grad in grads]

            tae = time.time() - start_time
            if (epoch + 1) % verbose_freq == 0 or epoch==100 or epoch==1:
                print(f'''Epoch:{epoch + 1}/{epochs} for Re {self.Re}''')
                for key, value in logs.items():
                    history[key].append(value.numpy())
                    print(f"{key}: {value:.4f} ", end="")
                print(f"Time: {tae / 60:.4f}min")

        return history

        
