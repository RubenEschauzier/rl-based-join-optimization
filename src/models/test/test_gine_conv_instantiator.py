from src.models.model_instantiator import ModelFactory

config_location = (r"C:\Users\ruben\projects\rl-based-join-optimization\experiments\model_configs\pretrain_model"
                   r"\t_cv_repr_exact_seperate_head.yaml")
model_factory_gine_conv= ModelFactory(config_location)
gine_conv_model = model_factory_gine_conv.load_gine_conv()
