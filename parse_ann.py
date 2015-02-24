
import sys
from array import array

''' FORMAT to parse

// last cost: 0.01969509199261665
// ANN
int sizes_size = 3;
int sizes[] = {10000,15,121,};
// biases
int bb_size = 136;
int bb[] = {0.008484145626425743,0.4636982381343842,0.9913022518157959,...,-0.9609371423721314,-0.2644922435283661,-0.7570790648460388,-1.059393644332886,};
// weights
int ww_size = 151815;
int ww[] = {0.522293746471405,0.4344615042209625,0.2652974128723145,...,-0.9928632974624634,-1.004094123840332,-1.639245510101318,-1.643823385238648,-1.650781631469727,};
// bb derivs
int bb_deriv_size = 136;
int bb_deriv[] = {-0.002991956658661366,-0.002405705396085978,-0.001411742647178471,...,0.004782990086823702,9.697333007352427e-005,6.180353537971195e-013,};
// ww derivs
int ww_deriv_size = 151815;
int ww_deriv[] = {0,0,0,0...,3.137185335729681e-013,4.501758395312333e-013,4.034915576569403e-013,};

'''

NO_DERIV = 0
WITH_DERIV = 1



def process(fname):

    # as floats
    a = array('f')

    fname_out = fname + '.b'

    with open(fname, "r") as fin:
        # 16 lines is expected
        cnt = 0
        for line in fin:
            cnt += 1
            line = line.strip()

            if line.startswith("// last cost: "):
                cost = float(line[14:])
            elif line.startswith("int sizes_size = "):
                sizes_size = float(line[len("int sizes_size = ") : -1])
            elif line.startswith("int sizes[] = {"):
                sizes = [float(v) for v in line[len("int sizes[] = {") : -3].split(",")]
            elif line.startswith("int bb_size = "):
                bb_size = float(line[len("int bb_size = ") : -1])
            elif line.startswith("int bb[] = {"):
                bb = [float(v) for v in line[len("int bb[] = {") : -3].split(",")]
            elif line.startswith("int ww_size = "):
                ww_size = float(line[len("int ww_size = ") : -1])
            elif line.startswith("int ww[] = {"):
                ww = [float(v) for v in line[len("int ww[] = {") : -3].split(",")]
            elif line.startswith("int bb_deriv_size = "):
                bb_deriv_size = float(line[len("int bb_deriv_size = ") : -1])
            elif line.startswith("int bb_deriv[] = {"):
                bb_deriv = [float(v) for v in line[len("int bb_deriv[] = {") : -3].split(",")]
            elif line.startswith("int ww_deriv_size = "):
                ww_deriv_size = float(line[len("int ww_deriv_size = ") : -1])
            elif line.startswith("int ww_deriv[] = {"):
                ww_deriv = [float(v) for v in line[len("int ww_deriv[] = {") : -3].split(",")]

        if cnt != 16:
            raise Exception("File format was changed, please make all necessary fixes before run")


        a.append(WITH_DERIV)
        a.append(cost)

        a.append(sizes_size)
        a.extend(sizes)

        a.append(bb_size)
        a.extend(bb)

        a.append(ww_size)
        a.extend(ww)

        a.append(bb_deriv_size)
        a.extend(bb_deriv)

        a.append(ww_deriv_size)
        a.extend(ww_deriv)

        fout = open(fname_out, "wb+")
        a.tofile(fout)
        fout.flush()
        fout.close()


    print "cost = %.17f" % cost
    print "sizes_size = %.0f" % sizes_size
    print "sizes = %s" % str(sizes)
    print "bb_size = %f" % bb_size
    print "bb = %s" % (str(bb[:3]) + "..." + str(bb[-3:])).replace("]...[", " ... ")
    print "ww_size = %f" % ww_size
    print "ww = %s" % (str(ww[:3]) + "..." + str(ww[-3:])).replace("]...[", " ... ")
    print "bb_deriv_size = %f" % bb_deriv_size
    print "bb_deriv = %s" % (str(bb_deriv[:3]) + "..." + str(bb_deriv[-3:])).replace("]...[", " ... ")
    print "ww_deriv_size = %f" % ww_deriv_size
    print "ww_deriv = %s" % (str(ww_deriv[:3]) + "..." + str(ww_deriv[-3:])).replace("]...[", " ... ")



def main():
    sys.argv.append(r'C:\Temp\kaggle\NDSB\tmp\ann_27_0.487815.b')
    fname = sys.argv[1]
    process(fname)


if __name__ == '__main__':
    main()
