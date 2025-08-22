import numpy as np

### revisited version of pos2dis and dis2pos functions that work with more than 3 channels ###
def pos2dis_(pos, boxsize, Ng):
    cellsize = boxsize / Ng
    lattice = np.arange(Ng) * cellsize + 0.5 * cellsize

    pos[..., 0] -= lattice.reshape(-1, 1, 1) 
    pos[..., 1] -= lattice.reshape(-1, 1)
    pos[..., 2] -= lattice

    # periodic boundary conditions on the first three channels only
    pos[..., :3] -= np.rint(pos[..., :3] / boxsize) * boxsize
    return pos


def dis2pos_(dis_field, boxsize, Ng):
    cellsize = boxsize / Ng
    lattice = np.arange(Ng) * cellsize + 0.5 * cellsize

    pos = dis_field.copy()

    pos[2] += lattice
    pos[1] += lattice.reshape(-1, 1)
    pos[0] += lattice.reshape(-1, 1, 1)

    # safe in-place wrapping
    pos[:3] = np.where(pos[:3] < 0, pos[:3] + boxsize, pos[:3])
    pos[:3] = np.where(pos[:3] > boxsize, pos[:3] - boxsize, pos[:3])

    return pos
##############################################################################################



def pos2dis(pos, boxsize, Ng):
    """Assume `pos` is ordered in `pid` that aligns with the Lagrangian lattice,
    and all displacement must not exceed half box size.
    """
    cellsize = boxsize / Ng
    lattice = np.arange(Ng) * cellsize + 0.5 * cellsize

    pos[..., 0] -= lattice.reshape(-1, 1, 1) 
    pos[..., 1] -= lattice.reshape(-1, 1)
    pos[..., 2] -= lattice

    pos -= np.rint(pos / boxsize) * boxsize #this line subtracts the boxsize from the position if the position is greater than boxsize

    return pos


def dis2pos(dis_field,boxsize,Ng):
    """Assume 'dis_field' is in order of `pid` that aligns with the Lagrangian lattice,
    and dis_field.shape = (3,Ng,Ng,Ng)
    dd"""
    cellsize = boxsize / Ng
    lattice = np.arange(Ng) * cellsize + 0.5 * cellsize

    pos = dis_field.copy()

    pos[2] += lattice
    pos[1] += lattice.reshape(-1, 1)
    pos[0] += lattice.reshape(-1, 1, 1)

    pos[pos<0] += boxsize
    pos[pos>boxsize] -= boxsize

    return pos


if __name__ == "__main__":
    # unit test for pos2dis and dis2pos
    for i in range(10):
        Ng = 32
        boxsize = 100.0
        pos_vel = np.random.rand(Ng, Ng, Ng, 6) * boxsize

        # offset some particles so that they are out of the box
        for i in range(Ng**3 // 10): #randomly offset 10% of the particles
            x = np.random.randint(0, Ng)
            y = np.random.randint(0, Ng)
            z = np.random.randint(0, Ng)
            pos_vel[x, y, z, :3] += boxsize * 1.5
        
        
        pos = pos_vel[..., :3]

        dis1 = pos2dis_(pos, boxsize, Ng) # new on crop
        dis2 = pos2dis(pos, boxsize, Ng) # old on crop

        dis_vel = pos2dis_(pos_vel, boxsize, Ng) 
        dis3 = dis_vel[..., :3] # new without crop

        assert np.allclose(dis1, dis2), "dis1 and dis2 are not equal!"
        assert np.allclose(dis1, dis3), "dis1 and dis3 are not equal!"
        

        print("Test 1 passed!")

        # now we convert dis_vel and dis1 back to positions and check if they are equal

        # make dis_vel and dis1 channel first
        dis_vel = np.moveaxis(dis_vel, -1, 0)
        dis1 = np.moveaxis(dis1, -1, 0)


        pos1 = dis2pos_(dis1, boxsize, Ng) # new on crop
        pos2 = dis2pos(dis1, boxsize, Ng) # old on crop
        pos3 = dis2pos_(dis_vel, boxsize, Ng) 
        pos3 = pos3[:3] # new without crop

        assert np.allclose(pos1, pos2), "pos1 and pos2 are not equal!"
        assert np.allclose(pos1, pos3), "pos1 and pos3 are not equal!"

        print("Test 2 passed!")


