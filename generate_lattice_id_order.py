#This script builds the original lattice and outputs the the flattened array of ids in a file. This array is used in preprocess.py to align the ids of the particles to properly compute displacement fields.
import numpy as np
import os
import argparse

def generate_initial_id_grid(Nmesh=1024, TileFac=4):
    Nbase = Nmesh // TileFac  # 1024 / 4 = 256

    ids = np.zeros((Nmesh, Nmesh, Nmesh), dtype=np.int64)

    IDStart = 1

    for i in range(TileFac):
        for j in range(TileFac):
            for k in range(TileFac):
                istart = i * Nbase
                jstart = j * Nbase
                kstart = k * Nbase

                for ii in range(Nbase):
                    for jj in range(Nbase):
                        for kk in range(Nbase):
                            ids[istart + ii, jstart + jj, kstart + kk] = IDStart
                            IDStart += 1

    return ids

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate id order that aligns with the Lagrangian lattice.')
    parser.add_argument('--Nmesh', type=int, default=1024, help='Number of divisions per side (default: 1024)')
    parser.add_argument('--TileFac', type=int, default=4, help='Tiling factor (default: 4)')
    parser.add_argument('--output', type=str, default='lattice_id_order.npy', help='Output file path (default: lattice_id_order.npy)')
    args = parser.parse_args()

    print(f"Nmesh: {args.Nmesh}")
    print(f"TileFac: {args.TileFac}")
    print(f"Output file: {args.output}")

    lattice_id_order = generate_initial_id_grid(Nmesh=args.Nmesh, TileFac=args.TileFac).flatten()

    print(f"Lattice id order shape: {lattice_id_order.shape}")
    print(f"First 10 ids: {lattice_id_order[:10]}")
    print(f"Last 10 ids: {lattice_id_order[-10:]}")

    np.save(args.output, lattice_id_order)
    print(f"Lattice id order saved to {args.output}")
    