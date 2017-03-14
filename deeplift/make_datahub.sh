#python make_datahub.py /srv/scratch/annashch/heterokaryon/make_tracks_heterokaryon_sorted/filtered/ --singledir
#python make_datahub.py 17/ --singledir
#for i in `seq 1 22` X Y 11_gl000202_random 1_gl000191_random  4_gl000194_random  Un_gl000219  Un_gl000224 17_gl000205_random 1_gl000192_random  7_gl000195_random  Un_gl000220  Un_gl000226
#for i in chr7_DEBUG
#for i in `seq 1 22` X Y 

#for i in `seq 1 22` X Y Un_gl000246 Un_gl000238 Un_gl000211 17_gl000203_random 19_gl000208_random Un_gl000239 17_gl000206_random Un_gl000249 Un_gl000235 Un_gl000212 Un_gl000222 8_gl000197_random 21_gl000210_random Un_gl000248 Un_gl000237 8_gl000196_random Un_gl000228 Un_gl000227 Un_gl000224 Un_gl000229 Un_gl000232 9_gl000198_random 11_gl000202_random Un_gl000219 Un_gl000242 Un_gl000231 Un_gl000223 4_gl000193_random Un_gl000241 17_gl000204_random Un_gl000234 9_gl000199_random 9_gl000201_random 1_gl000191_random Un_gl000216 1_gl000192_random 17_gl000205_random 7_gl000195_random Un_gl000225 Un_gl000220
#do
#python make_datahub.py old_deepLIFT_specialized_chr$i/ --singledir
#python make_datahub_dmso.py old_dmso_chr$i/ --singledir
#python make_datahub.py remapped_allpeaks_chr$i/ --singledir --trueAnnotationDir true_annotation_dir --predictedAnnotationDir predicted_annotation_dir #--motifDir motifGenomeLocations_chr$i
#python make_datahub_bigwig.py remapped_allpeaks_chr$i/ --singledir --trueAnnotationDir true_annotation_dir --predictedAnnotationDir predicted_annotation_dir #--motifDir motifGenomeLocations_chr$i
#python make_datahub_bigwig.py remapped_allpeaks_chr$i/ --singledir #--motifDir motifGenomeLocations_chr$i
#python make_datahub_bigwig.py debug.2kb.no_chr$i/ --singledir #--motifDir motifGenomeLocations_chr$i
#scp -r remapped_chr$i/*json annashch@mitra:/srv/www/kundaje/annashch/het/remapped_chr$i/
#echo $i
#done
#python make_datahub.py figure/ --singledir 


#python make_datahub.py oct4/ --singledir --trueAnnotationDir true_annotation_dir_OCT4 --predictedAnnotationDir predicted_annotation_dir_OCT4 #--motifDir motifGenomeLocations_chr$i
#python make_datahub_bigwig.py oct4/ --singledir --trueAnnotationDir true_annotation_dir_OCT4 --predictedAnnotationDir predicted_annotation_dir_OCT4 #--motifDir motifGenomeLocations_chr$i

#python make_datahub.py lin28a/ --singledir --trueAnnotationDir true_annotation_dir_LIN28A --predictedAnnotationDir predicted_annotation_dir_LIN28A #--motifDir motifGenomeLocations_chr$i
#python make_datahub_bigwig.py lin28a/ --singledir --trueAnnotationDir true_annotation_dir_LIN28A --predictedAnnotationDir predicted_annotation_dir_LIN28A #--motifDir motifGenomeLocations_chr$i

#updated set of figures 
#python make_datahub_bigwig_figures.py figures_UPDATED/ --singledir #--trueAnnotationDir true_annotation_dir_FIGURES --predictedAnnotationDir predicted_annotation_dir_FIGURES #--motifDir motifGenomeLocations_chr$i


#python make_datahub_bigwig.py dmso_NEWDATA/ --singledir --nameListFile task_names_dmso.txt #--trueAnnotationDir true_annotation_dir_FIGURES --predictedAnnotationDir predicted_annotation_dir_FIGURES #--motifDir motifGenomeLocations_chr$i

#python make_datahub_bigwig.py david.bassetlike.vars.full/ --singledir --nameListFile gecco.task.names.txt #--trueAnnotationDir true_annotation_dir_FIGURES --predictedAnnotationDir predicted_annotation_dir_FIGURES #--motifDir motifGenomeLocations_chr$i

python make_datahub_bigwig.py anna.positional.full/ --singledir --nameListFile gecco.task.names.txt 
