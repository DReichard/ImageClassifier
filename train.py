import prepare_dataset


clean_path = 'D:\\nir_datasets\\jpg\\clean\\clean_cut_images1'
affected_path = 'D:\\nir_datasets\\jpg\\affected\\sorted\\vsl_lsb_5kb\\2019-03-17-15-04-40-207\\input000';
limit = 5000

images, labels = prepare_dataset.get_dataset(clean_path, affected_path, limit)

print(images.shape)