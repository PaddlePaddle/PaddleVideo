file=$1
while read line
do
    wget "$line"
    array=(${line//// })
    file_name=${array[-1]}
    cmd=`tar -xzvf ${file_name} --strip-components 1 -C ./videos`
    eval $cmd
done <$file
