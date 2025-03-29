from happyrec.models.multi_task import SharedBottom


def shared_task_layers(model):

    shared_layers = [model.embedding.trainable_variables]
    task_layers = []
    
    # TODO：embedding 层？
    if isinstance(model, SharedBottom):
        shared_layers.extend(model.bottom_mlp.trainable_variables)
        # 任务层包含 towers 和 prediction
        task_layers.extend(model.towers.trainable_variables)
        task_layers.extend(model.predict_layers.trainable_variables)
    
    # elif isinstance(model, MMOE):
    #     # MMOE 结构：experts为共享层
    #     shared_vars.extend(model.experts.trainable_variables)
    #     # 任务层包含 gates + towers + prediction
    #     task_vars.extend(model.gates.trainable_variables)
    #     task_vars.extend(model.towers.trainable_variables)
    #     task_vars.extend(model.predict_layers.trainable_variables)
    
    # elif isinstance(model, PLE):
    #     # PLE 结构：cgc_layers为共享层
    #     shared_vars.extend(model.cgc_layers.trainable_variables)
    #     # 任务层包含 towers + prediction
    #     task_vars.extend(model.towers.trainable_variables)
    #     task_vars.extend(model.predict_layers.trainable_variables)
    
    # elif isinstance(model, AITM):
    #     # AITM 结构：bottoms为共享层
    #     shared_vars.extend(model.bottoms.trainable_variables)
    #     # 任务层包含 info_gates + towers + aits
    #     task_vars.extend(model.info_gates.trainable_variables)
    #     task_vars.extend(model.towers.trainable_variables)
    #     task_vars.extend(model.aits.trainable_variables)
    
    else:
        raise ValueError(f'Model {model.__class__.__name__} not supported for MetaBalance')
    
    return shared_layers, task_layers