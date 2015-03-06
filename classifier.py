

import os
import sys
import numpy as np
import scipy as sp
from array import array
import datetime as dt
import ctypes

sys.path.append("..\\")
import pathes


MAX_ROWS = 48
MAX_COLS = 48

VEC_LEN = 1 + 2 + MAX_ROWS * MAX_COLS

ANN_DLL = ctypes.cdll.LoadLibrary("ann.dll")


title = 'image,acantharia_protist_big_center,acantharia_protist_halo,acantharia_protist,amphipods,appendicularian_fritillaridae,appendicularian_s_shape,appendicularian_slight_curve,appendicularian_straight,artifacts_edge,artifacts,chaetognath_non_sagitta,chaetognath_other,chaetognath_sagitta,chordate_type1,copepod_calanoid_eggs,copepod_calanoid_eucalanus,copepod_calanoid_flatheads,copepod_calanoid_frillyAntennae,copepod_calanoid_large_side_antennatucked,copepod_calanoid_large,copepod_calanoid_octomoms,copepod_calanoid_small_longantennae,copepod_calanoid,copepod_cyclopoid_copilia,copepod_cyclopoid_oithona_eggs,copepod_cyclopoid_oithona,copepod_other,crustacean_other,ctenophore_cestid,ctenophore_cydippid_no_tentacles,ctenophore_cydippid_tentacles,ctenophore_lobate,decapods,detritus_blob,detritus_filamentous,detritus_other,diatom_chain_string,diatom_chain_tube,echinoderm_larva_pluteus_brittlestar,echinoderm_larva_pluteus_early,echinoderm_larva_pluteus_typeC,echinoderm_larva_pluteus_urchin,echinoderm_larva_seastar_bipinnaria,echinoderm_larva_seastar_brachiolaria,echinoderm_seacucumber_auricularia_larva,echinopluteus,ephyra,euphausiids_young,euphausiids,fecal_pellet,fish_larvae_deep_body,fish_larvae_leptocephali,fish_larvae_medium_body,fish_larvae_myctophids,fish_larvae_thin_body,fish_larvae_very_thin_body,heteropod,hydromedusae_aglaura,hydromedusae_bell_and_tentacles,hydromedusae_h15,hydromedusae_haliscera_small_sideview,hydromedusae_haliscera,hydromedusae_liriope,hydromedusae_narco_dark,hydromedusae_narco_young,hydromedusae_narcomedusae,hydromedusae_other,hydromedusae_partial_dark,hydromedusae_shapeA_sideview_small,hydromedusae_shapeA,hydromedusae_shapeB,hydromedusae_sideview_big,hydromedusae_solmaris,hydromedusae_solmundella,hydromedusae_typeD_bell_and_tentacles,hydromedusae_typeD,hydromedusae_typeE,hydromedusae_typeF,invertebrate_larvae_other_A,invertebrate_larvae_other_B,jellies_tentacles,polychaete,protist_dark_center,protist_fuzzy_olive,protist_noctiluca,protist_other,protist_star,pteropod_butterfly,pteropod_theco_dev_seq,pteropod_triangle,radiolarian_chain,radiolarian_colony,shrimp_caridean,shrimp_sergestidae,shrimp_zoea,shrimp-like_other,siphonophore_calycophoran_abylidae,siphonophore_calycophoran_rocketship_adult,siphonophore_calycophoran_rocketship_young,siphonophore_calycophoran_sphaeronectes_stem,siphonophore_calycophoran_sphaeronectes_young,siphonophore_calycophoran_sphaeronectes,siphonophore_other_parts,siphonophore_partial,siphonophore_physonect_young,siphonophore_physonect,stomatopod,tornaria_acorn_worm_larvae,trichodesmium_bowtie,trichodesmium_multiple,trichodesmium_puff,trichodesmium_tuft,trochophore_larvae,tunicate_doliolid_nurse,tunicate_doliolid,tunicate_partial,tunicate_salp_chains,tunicate_salp,unknown_blobs_and_smudges,unknown_sticks,unknown_unclassified'
classes_to_title_map = {'image' : 0, 'acantharia_protist_big_center' : 1, 'acantharia_protist_halo' : 2, 'acantharia_protist' : 3, 'amphipods' : 4, 'appendicularian_fritillaridae' : 5, 'appendicularian_s_shape' : 6, 'appendicularian_slight_curve' : 7, 'appendicularian_straight' : 8, 'artifacts_edge' : 9, 'artifacts' : 10, 'chaetognath_non_sagitta' : 11, 'chaetognath_other' : 12, 'chaetognath_sagitta' : 13, 'chordate_type1' : 14, 'copepod_calanoid_eggs' : 15, 'copepod_calanoid_eucalanus' : 16, 'copepod_calanoid_flatheads' : 17, 'copepod_calanoid_frillyAntennae' : 18, 'copepod_calanoid_large_side_antennatucked' : 19, 'copepod_calanoid_large' : 20, 'copepod_calanoid_octomoms' : 21, 'copepod_calanoid_small_longantennae' : 22, 'copepod_calanoid' : 23, 'copepod_cyclopoid_copilia' : 24, 'copepod_cyclopoid_oithona_eggs' : 25, 'copepod_cyclopoid_oithona' : 26, 'copepod_other' : 27, 'crustacean_other' : 28, 'ctenophore_cestid' : 29, 'ctenophore_cydippid_no_tentacles' : 30, 'ctenophore_cydippid_tentacles' : 31, 'ctenophore_lobate' : 32, 'decapods' : 33, 'detritus_blob' : 34, 'detritus_filamentous' : 35, 'detritus_other' : 36, 'diatom_chain_string' : 37, 'diatom_chain_tube' : 38, 'echinoderm_larva_pluteus_brittlestar' : 39, 'echinoderm_larva_pluteus_early' : 40, 'echinoderm_larva_pluteus_typeC' : 41, 'echinoderm_larva_pluteus_urchin' : 42, 'echinoderm_larva_seastar_bipinnaria' : 43, 'echinoderm_larva_seastar_brachiolaria' : 44, 'echinoderm_seacucumber_auricularia_larva' : 45, 'echinopluteus' : 46, 'ephyra' : 47, 'euphausiids_young' : 48, 'euphausiids' : 49, 'fecal_pellet' : 50, 'fish_larvae_deep_body' : 51, 'fish_larvae_leptocephali' : 52, 'fish_larvae_medium_body' : 53, 'fish_larvae_myctophids' : 54, 'fish_larvae_thin_body' : 55, 'fish_larvae_very_thin_body' : 56, 'heteropod' : 57, 'hydromedusae_aglaura' : 58, 'hydromedusae_bell_and_tentacles' : 59, 'hydromedusae_h15' : 60, 'hydromedusae_haliscera_small_sideview' : 61, 'hydromedusae_haliscera' : 62, 'hydromedusae_liriope' : 63, 'hydromedusae_narco_dark' : 64, 'hydromedusae_narco_young' : 65, 'hydromedusae_narcomedusae' : 66, 'hydromedusae_other' : 67, 'hydromedusae_partial_dark' : 68, 'hydromedusae_shapeA_sideview_small' : 69, 'hydromedusae_shapeA' : 70, 'hydromedusae_shapeB' : 71, 'hydromedusae_sideview_big' : 72, 'hydromedusae_solmaris' : 73, 'hydromedusae_solmundella' : 74, 'hydromedusae_typeD_bell_and_tentacles' : 75, 'hydromedusae_typeD' : 76, 'hydromedusae_typeE' : 77, 'hydromedusae_typeF' : 78, 'invertebrate_larvae_other_A' : 79, 'invertebrate_larvae_other_B' : 80, 'jellies_tentacles' : 81, 'polychaete' : 82, 'protist_dark_center' : 83, 'protist_fuzzy_olive' : 84, 'protist_noctiluca' : 85, 'protist_other' : 86, 'protist_star' : 87, 'pteropod_butterfly' : 88, 'pteropod_theco_dev_seq' : 89, 'pteropod_triangle' : 90, 'radiolarian_chain' : 91, 'radiolarian_colony' : 92, 'shrimp_caridean' : 93, 'shrimp_sergestidae' : 94, 'shrimp_zoea' : 95, 'shrimp-like_other' : 96, 'siphonophore_calycophoran_abylidae' : 97, 'siphonophore_calycophoran_rocketship_adult' : 98, 'siphonophore_calycophoran_rocketship_young' : 99, 'siphonophore_calycophoran_sphaeronectes_stem' : 100, 'siphonophore_calycophoran_sphaeronectes_young' : 101, 'siphonophore_calycophoran_sphaeronectes' : 102, 'siphonophore_other_parts' : 103, 'siphonophore_partial' : 104, 'siphonophore_physonect_young' : 105, 'siphonophore_physonect' : 106, 'stomatopod' : 107, 'tornaria_acorn_worm_larvae' : 108, 'trichodesmium_bowtie' : 109, 'trichodesmium_multiple' : 110, 'trichodesmium_puff' : 111, 'trichodesmium_tuft' : 112, 'trochophore_larvae' : 113, 'tunicate_doliolid_nurse' : 114, 'tunicate_doliolid' : 115, 'tunicate_partial' : 116, 'tunicate_salp_chains' : 117, 'tunicate_salp' : 118, 'unknown_blobs_and_smudges' : 119, 'unknown_sticks' : 120, 'unknown_unclassified' : 121}


ann_fnames = [
("trichodesmium_puff_0.150171_1999.b.b", 110),
("chaetognath_other_0.325235_11999.b.b", 11),
("copepod_cyclopoid_oithona_eggs_0.150951_11225.b.b", 25),

("protist_other_0.353022_599.b.b", 85),       # protist_other
("detritus_other_0.305694_599.b.b", 35),       # detritus_other
(None, 24),       # copepod_cyclopoid_oithona
(None, None),       # acantharia_protist
(None, None),       # chaetognath_non_sagitta
(None, None),       # trichodesmium_bowtie
(None, None),       # hydromedusae_solmaris
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),
(None, None),       #
(None,  1),       # acantharia_protist_big_center
(None, 66),       # hydromedusae_other
(None, 56),       # heteropod
(None, 50),       # fish_larvae_deep_body
(None, 61)        # hydromedusae_haliscera_small_sideview
]


anns = [None] * 121



def load_classes():
    res = {}
    with open("..\\..\\data\\classes.txt", "r") as fin:
        for line in fin:
            line = line.strip()
            tokens = line.split(" ")
            # ID to name map
            res[int(tokens[0])] = tokens[1]
    return res


def process():

    classes = load_classes()

    # init
    for i in range(121):
        fname = ann_fnames[i][0]
        if None != fname:
            fname = pathes.path_base + "tmp\\" + fname
            anns[i] = ANN_DLL.ann_fromfile(ctypes.c_char_p(fname))
        else:
            break

    #
    path = pathes.path_data + "test_ann1\\"

    predictions = np.array([0] * 2, dtype=np.float32)

    cnt = 0

    with open(pathes.path_base + "submission.txt", "w+") as fout:
        fout.write("%s%s" % (title, os.linesep))

        for f in os.listdir(path):
            orig_fname = f[:-2]

            fname = path + f
            with open(fname, "rb") as fin:
                x = array('f')
                x.fromfile(fin, VEC_LEN)
                x = x[1:]

                cls_name = None

                for i in range(121):
                    if None != anns[i]:
                        ANN_DLL.ann_predict(ctypes.c_void_p(anns[i]), ctypes.c_void_p(x.buffer_info()[0]), ctypes.c_void_p(predictions.ctypes.data), ctypes.c_int(1))

                        if predictions[0] > .9:
                            cls_name = classes[ann_fnames[i][1]]
                            break
                    else:
                        predictions[0] = -1.
                        break

#                if cls_name == None:
#                    cls_name = "unknown_unclassified"

                index = -1
                if None != cls_name:
                    index = classes_to_title_map[cls_name] - 1

                fout.write("%s" % orig_fname)
                for i in range(121):
                    if i == index:
                        fout.write(",1")
                    elif -1 == index:
                        fout.write(",0.00826446")
                    else:
                        fout.write(",0")
                fout.write("%s" % os.linesep)
                fout.flush()

            cnt += 1
            if 0 == (cnt % 1000):
                print "# processed ", cnt, "files"


def main():
    process()

if __name__ == '__main__':
    main()




