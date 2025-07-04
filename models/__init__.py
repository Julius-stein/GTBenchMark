from models.GT import GT

def GraphTransformer(net_params):
    return GT(net_params)

def gt_model(MODEL_NAME, net_params):
    models = {
        'GraphTransformer': GraphTransformer
    }
        
    return models[MODEL_NAME](net_params)