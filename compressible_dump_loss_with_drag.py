#!/home/cooz/anaconda3/bin/python

import numpy as np
import argparse
import sys
import scipy.optimize as scopt

def main():
    intro("Compressible Sudden Expansion with Drag")

    args = get_args()

    regurgitate(args)

    m2,pr,ptloss,cploss = momentum_and_continuity(args)

    print("{0:20s}{1:10.6f}\n".format("Exit Mach No. ",np.float(m2)))
    print("{0:20s}{1:10.6f}\n".format("Pt2/Pt1",np.float(pr)))
    print("{0:20s}{1:10.6f}\n".format("Pt Loss",np.float(ptloss)))
    print("{0:20s}{1:10.6f}\n".format("Cp Loss",np.float(cploss)))

################################################################################
# subroutines and functions

def intro(string):
    print("\n\n",end='',sep='')
    print("+","-"*78,"+",end="\n",sep='')
    print("|",str(string).center(78,' '),"|",end="\n",sep='')
    print("+","-"*78,"+",end="\n\n",sep='')
    sys.stdout.flush()


def get_args():
    parser = argparse.ArgumentParser(description="Compressible Sudden Expansion with Drag Force. Assumes constant specific heats, subsonic, compressible, non-swirling flow")
    parser.add_argument('--area-ratio','-ar',default=None,type=np.float,dest='ar',help="Exit Area / Inlet Area")
    parser.add_argument('--drag-area-ratio','-ad',default=None,type=np.float,dest='ad',help='Drag Area / Inlet Area')
    parser.add_argument('-cd',default=None,type=np.float,dest='cd',help='Drag Coefficient')
    parser.add_argument('--mach1','-m1',default=None,type=np.float,dest='m1',help="Inlet Mach Number")
    parser.add_argument('--cp/cv','-k',default=np.float(1.4),type=np.float,dest='k',help="Ratio of Specific Heats")

    args = parser.parse_args()
    check_args(args)

    return args

def check_args(args):
    if( args.ar == None):
        print("{:<20s}{:<60s}".format("ERROR","Area Ratio not given\n\n"))
        sys.exit(1)
    if( args.m1 == None):
        print("{:<20s}{:<60s}".format("ERROR","Inlet Mach Number not given\n\n"))
        sys.exit(1)
    if( args.ad == None):
        print("{:<20s}{:<60s}".format("ERROR","Drag Area Ratio not given\n\n"))
        sys.exit(1)

def regurgitate(args):
    print("Inputs\n","-"*20,end="\n",sep='')
    print("{0:20s}{1:>10.6f}".format("Area Ratio",np.float(args.ar)),end="\n",sep='')
    print("{0:20s}{1:>10.6f}".format("Drag Area Ratio",np.float(args.ad)),end="\n",sep='')
    print("{0:20s}{1:>10.6f}".format("Drag Coefficient",np.float(args.cd)),end="\n",sep='')
    print("{0:20s}{1:>10.6f}".format("Inlet Mach No",np.float(args.m1)),end="\n",sep='')
    print("{0:20s}{1:>10.6f}".format("Cp/Cv",np.float(args.k)),end="\n",sep='')
    print("\n\n",end='',sep='')


def momentum_and_continuity(args):
    """ Solve momentum and continuity for sudden expansion with a drag force.
    Energy is satisfied in the derivation of the continuity equation.
    """
    m1 = args.m1
    k  = args.k
    ar = args.ar
    ad = args.ad
    cd = args.cd

    def momentum_and_continuity(initialization):

        m2,pr = initialization
        fmn1 = np.float( 1. + (k-1.)/2. * m1**2.)
        fmn2 = np.float( 1. + (k-1.)/2. * m2**2.)

        continuity = pr * m2/m1 * ar * (fmn2/fmn1)**(-(k+1.0)/(2*(k-1.0))) - 1.0
        momentum   = pr * ar * fmn2**(-k/(k-1.0)) * (1.0 + k * m2**2.0) \
                - fmn1**(-k/(k-1.0)) * (ar + k * m1**2.0) \
                + cd * ad * (1.0-fmn1**(-k/(k-1.0)))

        return (continuity,momentum)

    initialization = np.array([m1,0.9])
    solution,infodict,ier,mesg =scopt.fsolve(momentum_and_continuity,initialization,full_output=True)

    m2, pr = solution

    if( ier == True):
        print('Converged in {0:3d} iterations'.format(infodict['nfev']),end="\n\n")
    else:
        print("\n\nERROR Did not converge!",sep='',end="\n\n")
        sys.exit(1)

    ptloss = 1.0 - pr
    fmn1 = np.float( 1. + (k-1.)/2. * m1**2.)
    cploss = ptloss/(1.0 - fmn1**(-k/(k-1.0)))

    return m2,pr,ptloss,cploss


if __name__ == '__main__':
    main()
