#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <set>
#include <array>
#include <omp.h>
#include <math.h>

using namespace std;

struct dim3
{
    dim3(int _x) { x = _x; y = z = 1; }
    dim3(int _x, int _y) { x = _x; y = _y; z = 1; }
    dim3(int _x, int _y, int _z) { x = _x; y = _y; z = _z; }
    dim3() { x = y = z = 1; }
    int x, y, z;
};

//ii j i
int lowI[3] = { 3, 6, 3 };
int highI[3] = { 5, 3, 7 };

int idxI[3] = { 1, -1, 1 };

set<array<int, 3>> elems;

static void kernel(int id_x, int id_y,
                    int base_i, int base_j, int base_ii,
                    int step_i, int step_j, int step_ii,
                    int max_z, int SE, int var1, int var2, int var3,
                    int Emax, int Emin, int min_ij, int swap_ij,
                    int type_of_run, int idxs_0, int idxs_1, int idxs_2)
{
    int coords[3];

    // Local needs
    int ii, j, i;
    //id_x = x;// blockIdx.x* blockDim.x + threadIdx.x;
    //id_y = y;// blockIdx.y* blockDim.y + threadIdx.y;
    if (id_y < max_z)
    {
        if (id_y + SE < Emin)
            i = id_y + SE;
        else
        {
            if (id_y + SE < Emax)
                i = min_ij;
            else
                i = 2 * min_ij - SE - id_y + Emax - Emin - 1;
        }

        if (id_x < i)
        {
            if (var3 == 1 && Emin < id_y + SE)
            {
                base_i = base_i - step_i * (SE + id_y - Emin);
                base_j = base_j + step_j * (SE + id_y - Emin);
            }

            coords[idxs_0] = base_i + (id_y * (var1 + var3) - id_x) * step_i;
            coords[idxs_1] = base_j + (id_y * var2 + id_x) * step_j;
            coords[idxs_2] = base_ii - id_y * step_ii;

            if (swap_ij * var3)
                coords[idxs_0] ^= coords[idxs_1] ^= coords[idxs_0] ^= coords[idxs_1];

            i = coords[0];
            j = coords[1];
            ii = coords[2];

            if ((i < lowI[2] || i > highI[2]) && idxI[2] > 0 ||
                (i > lowI[2] || i < highI[2]) && idxI[2] < 0)
            {
                printf("error on I\n");
                exit(-1);
            }
            if ((j < lowI[1] || j > highI[1]) && idxI[1] > 0 ||
                (j > lowI[1] || j < highI[1]) && idxI[1] < 0)
            {
                printf("error on J\n");
                exit(-1);
            }
            if ((ii < lowI[0] || ii > highI[0]) && idxI[0] > 0 ||
                (ii > lowI[0] || ii < highI[0]) && idxI[0] < 0)
            {
                printf("error on II\n");
                exit(-1);
            }
            // Loop body
            /*printf("[%d %d %d] | %d %d %d %d %d %d | %d %d | %d %d %d | %d %d %d %d| %d %d %d %d|\n", i, j, ii,
                base_i, base_j, base_ii, step_i, step_j, step_ii,
                max_z, SE, var1, var2, var3, Emax, Emin, min_ij, swap_ij,
                type_of_run, idxs_0, idxs_1, idxs_2);*/

            array<int, 3> next = { i, j, ii };
            if (elems.find(next) != elems.end())
            {
                printf("error on elems\n");
                exit(-1);
            }
            else
                elems.insert(next);
        }
    }
}

static void loop_kernel(const dim3& blocks, const dim3& threads, 
                        int base_i, int base_j, int base_ii,
                        int step_i, int step_j, int step_ii, 
                        int max_z, int SE, int var1, int var2, int var3, 
                        int Emax, int Emin, int min_ij, int swap_ij, 
                        int type_of_run, int idxs_0, int idxs_1, int idxs_2)
{
    for (int y = 0; y < blocks.y * threads.y; ++y)
        for (int x = 0; x < blocks.x * threads.x; ++x)  
            kernel(x, y, base_i, base_j, base_ii, step_i, step_j, step_ii,
                   max_z, SE, var1, var2, var3, Emax, Emin, min_ij, swap_ij,
                   type_of_run, idxs_0, idxs_1, idxs_2);
}

void testAcross_7case()
{
    dim3 blocks, threads;
    int base_i, base_j, base_ii;
    int var3 = 0;
    int var2 = 0;
    int var1 = 1;
    int diag = 1;
    int SE = 1;
    int Emax, Emin, Allmin;

    int num_y;
    int num_x;

    int idxs[5] = { 0, 1, 2 };

    int lowI[3];
    int highI[3];
    int idxI[3];
    for (int k = 0; k < 3; ++k)
    {
        lowI[k] = ::lowI[k];
        highI[k] = ::highI[k];
        idxI[k] = ::idxI[k];
    }

    threads = dim3(8, 4, 1);
    num_x = threads.x;
    num_y = threads.y;

    const int Mi = (abs(lowI[2] - highI[2]) + 1) / abs(idxI[2]) + ((abs(lowI[2] - highI[2]) + 1) % abs(idxI[2]) != 0);
    const int Mj = (abs(lowI[1] - highI[1]) + 1) / abs(idxI[1]) + ((abs(lowI[1] - highI[1]) + 1) % abs(idxI[1]) != 0);
    const int Mk = (abs(lowI[0] - highI[0]) + 1) / abs(idxI[0]) + ((abs(lowI[0] - highI[0]) + 1) % abs(idxI[0]) != 0);
    Allmin = std::min(std::min(Mi, Mj), Mk);
    Emin = std::min(Mi, Mj);
    Emax = std::min(Mi, Mj) + abs(Mi - Mj) + 1;
    blocks = dim3(num_x, num_y);

    // Start method
    base_i = lowI[2];
    base_j = lowI[1];
    base_ii = lowI[0];
    int type_of_run = 7;
    while (diag <= Allmin)
    {
        blocks.x = diag / num_x + (diag % num_x != 0);
        blocks.y = diag / num_y + (diag % num_y != 0);
        loop_kernel(blocks, threads, base_i, base_j, base_ii, idxI[2], idxI[1], idxI[0], diag, SE, var1, var2, var3, Emax, Emin,
            std::min(Mi, Mj), Mi > Mj, type_of_run, idxs[0], idxs[1], idxs[2]);

        //printf("1===========\n");
        base_ii = base_ii + idxI[0];
        diag = diag + 1;
    }
    var1 = 0;
    var2 = 0;
    var3 = 1;

    if (Mk > Emin)
    {
        base_i = lowI[2] * (Mi <= Mj) + lowI[1] * (Mi > Mj);
        base_j = lowI[1] * (Mi <= Mj) + lowI[2] * (Mi > Mj);
        diag = Allmin + 1;

        while (diag - 1 != Mk)
        {
            blocks.x = Emin / num_x + (Emin % num_x != 0);
            blocks.y = diag / num_y + (diag % num_y != 0);
            if (Mi > Mj)
                idxI[2] ^= idxI[1] ^= idxI[2] ^= idxI[1];
            loop_kernel(blocks, threads, base_i, base_j, base_ii, idxI[2], idxI[1], idxI[0], diag, SE, var1, var2, var3, Emax, Emin,
                std::min(Mi, Mj), Mi > Mj, type_of_run, idxs[0], idxs[1], idxs[2]);
            if (Mi > Mj)
                idxI[2] ^= idxI[1] ^= idxI[2] ^= idxI[1];
            //printf("2===========\n");
            base_ii = base_ii + idxI[0];
            diag = diag + 1;
        }
    }
    diag = Mk;
    blocks.y = diag / num_y + (diag % num_y != 0);
    blocks.x = Emin / num_x + (Emin % num_x != 0);
    SE = 2;
    base_i = (lowI[2] + idxI[2]) * (Mi <= Mj) + (lowI[1] + idxI[1]) * (Mi > Mj);
    base_j = lowI[1] * (Mi <= Mj) + lowI[2] * (Mi > Mj);
    base_ii = lowI[0] + idxI[0] * (Mk - 1);

    while (Mi + Mj - Allmin != SE - 1)
    {
        if (Mi > Mj)
            idxI[2] ^= idxI[1] ^= idxI[2] ^= idxI[1];
        loop_kernel(blocks, threads, base_i, base_j, base_ii, idxI[2], idxI[1], idxI[0], diag, SE, var1, var2, var3, Emax, Emin,
            std::min(Mi, Mj), Mi > Mj, type_of_run, idxs[0], idxs[1], idxs[2]);
        if (Mi > Mj)
            idxI[2] ^= idxI[1] ^= idxI[2] ^= idxI[1];

        //printf("3===========\n");
        base_i = base_i + idxI[2] * (Mi <= Mj) + idxI[1] * (Mi > Mj);
        SE = SE + 1;
    }

    var1 = 0;
    var2 = 1;
    var3 = 0;
    diag = Allmin - 1;
    base_i = lowI[2] + idxI[2] * (Mi - 1);
    base_j = lowI[1] * (Mi > Mj) + base_j * (Mi <= Mj);
    if (Mi > Mj && Mk <= Emin)
    {
        base_j = base_j + idxI[1] + abs(Emin - Mk) * (idxI[1] > 0 ? 1 : -1);
    }
    else
    {
        if (Mi <= Mj && Mk <= Emin)
        {
            if (idxI[1] > 0)
            {
                base_j = base_j + idxI[1] + Emax - Emin - 1 + abs(Emin - Mk);
            }
            else
            {
                base_j = base_j + idxI[1] - Emax + Emin + 1 + Mk - Emin;
            }
        }
        else
        {
            if (Mi > Mj && Mk > Emin)
            {
                base_j = base_j + idxI[1];
            }
            else
            {
                if (Mi <= Mj && Mk > Emin)
                {
                    if (idxI[1] > 0)
                    {
                        base_j = base_j + idxI[1] + Emax - Emin - 1;
                    }
                    else
                    {
                        base_j = base_j + idxI[1] - Emax + Emin + 1;
                    }
                }
            }
        }
    }

    while (diag != 0)
    {
        blocks.x = diag / num_x + (diag % num_x != 0);
        blocks.y = diag / num_y + (diag % num_y != 0);
        loop_kernel(blocks, threads, base_i, base_j, base_ii, idxI[2], idxI[1], idxI[0], diag, SE, var1, var2, var3, Emax, Emin,
            std::min(Mi, Mj), Mi > Mj, type_of_run, idxs[0], idxs[1], idxs[2]);

        //printf("4===========\n");
        SE = SE + 1;
        base_j = base_j + idxI[1];
        diag = diag - 1;
    }

    if ((int)elems.size() != (abs(highI[2] - lowI[2]) + 1) * (abs(highI[1] - lowI[1]) + 1) * (abs(highI[0] - lowI[0]) + 1))
    {
        printf(" elems count = %d, total %d\n", (int)elems.size(), (abs(highI[2] - lowI[2]) + 1) * (abs(highI[1] - lowI[1]) + 1) * (abs(highI[0] - lowI[0]) + 1));
        exit(-2);
    }
}

int main()
{
    testAcross_7case();
        
    for (int z = 1; z < 10; ++z)
    {
        for (int k = 1; k < 10; ++k)
        {
            for (int j = 1; j < 10; ++j)
            {
                lowI[0] = 1;
                lowI[1] = 1;
                lowI[2] = 1;

                highI[0] = j + 1;
                highI[1] = k + 1;
                highI[2] = z + 1;

                idxI[0] = 1;
                idxI[1] = 1;
                idxI[2] = 1;

                elems.clear();
                testAcross_7case();
            }
        }
    }
    printf("done full +\n");

    for (int z = 1; z < 10; ++z)
    {
        for (int k = 1; k < 10; ++k)
        {
            for (int j = 1; j < 10; ++j)
            {
                lowI[0] = 1;
                lowI[1] = 1;
                lowI[2] = z + 1;

                highI[0] = j + 1;
                highI[1] = k + 1;
                highI[2] = 1;

                idxI[0] = 1;
                idxI[1] = 1;
                idxI[2] = -1;

                elems.clear();
                testAcross_7case();
            }
        }
    }
    printf("done - last\n");

    for (int z = 1; z < 10; ++z)
    {
        for (int k = 1; k < 10; ++k)
        {
            for (int j = 1; j < 10; ++j)
            {
                lowI[0] = 1;
                lowI[1] = k + 1;
                lowI[2] = 1;

                highI[0] = j + 1;
                highI[1] = 1;
                highI[2] = z + 1;

                idxI[0] = 1;
                idxI[1] = -1;
                idxI[2] = 1;

                elems.clear();
                testAcross_7case();
            }
        }
    }
    printf("done - mid\n");

    for (int z = 1; z < 10; ++z)
    {
        for (int k = 1; k < 10; ++k)
        {
            for (int j = 1; j < 10; ++j)
            {
                lowI[0] = j + 1;
                lowI[1] = 1;
                lowI[2] = 1;

                highI[0] = 1;
                highI[1] = k + 1;
                highI[2] = z + 1;

                idxI[0] = -1;
                idxI[1] = 1;
                idxI[2] = 1;

                elems.clear();
                testAcross_7case();
            }
        }
    }
    printf("done - first\n");

    for (int z = 1; z < 10; ++z)
    {
        for (int k = 1; k < 10; ++k)
        {
            for (int j = 1; j < 10; ++j)
            {
                lowI[0] = 1;
                lowI[1] = k + 1;
                lowI[2] = z + 1;

                highI[0] = j + 1;
                highI[1] = 1;
                highI[2] = 1;

                idxI[0] = 1;
                idxI[1] = -1;
                idxI[2] = -1;

                elems.clear();
                testAcross_7case();
            }
        }
    }
    printf("done - mid last\n");

    for (int z = 1; z < 10; ++z)
    {
        for (int k = 1; k < 10; ++k)
        {
            for (int j = 1; j < 10; ++j)
            {
                lowI[0] = j + 1;
                lowI[1] = k + 1;
                lowI[2] = 1;

                highI[0] = 1;
                highI[1] = 1;
                highI[2] = z + 1;

                idxI[0] = -1;
                idxI[1] = -1;
                idxI[2] = 1;

                elems.clear();
                testAcross_7case();
            }
        }
    }
    printf("done - first mid\n");

    for (int z = 1; z < 10; ++z)
    {
        for (int k = 1; k < 10; ++k)
        {
            for (int j = 1; j < 10; ++j)
            {
                lowI[0] = j + 1;
                lowI[1] = 1;
                lowI[2] = z + 1;

                highI[0] = 1;
                highI[1] = k + 1;
                highI[2] = 1;

                idxI[0] = -1;
                idxI[1] = 1;
                idxI[2] = -1;

                elems.clear();
                testAcross_7case();
            }
        }
    }
    printf("done - first last \n");

    for (int z = 1; z < 10; ++z)
    {
        for (int k = 1; k < 10; ++k)
        {
            for (int j = 1; j < 10; ++j)
            {
                lowI[0] = j + 1;
                lowI[1] = k + 1;
                lowI[2] = z + 1;

                highI[0] = 1;
                highI[1] = 1;
                highI[2] = 1;

                idxI[0] = -1;
                idxI[1] = -1;
                idxI[2] = -1;

                elems.clear();
                testAcross_7case();
            }
        }
    }
    printf("done full -\n");
    return 0;
}
