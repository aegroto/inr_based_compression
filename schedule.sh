for experiment_id in "basic_celeba" "maml_celeba"
do
    for config_path in exp/$experiment_id/*
    do
        config_id=$(basename $config_path)
        for image_path in data/CelebA100/*.png
        do
            image_name=$(basename $image_path)
            image_id=${image_name%.png}
            ./extract_stats.sh $experiment_id $config_id $image_id CelebA100
        done
    done
done

for experiment_id in "maml_kodak"
do
    for config_path in exp/$experiment_id/*
    do
        config_id=$(basename $config_path)
        for image_path in data/KODAK/*.png
        do
            image_name=$(basename $image_path)
            image_id=${image_name%.png}
            ./extract_stats.sh $experiment_id $config_id $image_id KODAK
        done
    done
done
