from paddlevideo.utils import get_logger
import paddle

def apply_to_static(config, model):
    logger = get_logger("paddlevideo")
    support_to_static = config.get('TO_STATIC', False)
    if not support_to_static:
        return model

    enable_to_static = config['TO_STATIC'].get("enable_to_static", False)
    if enable_to_static:
        specs = None #create_input_specs()
        is_pass = config['TO_STATIC'].get('enable_pass', False)
        if is_pass:
            build_strategy = paddle.static.BuildStrategy()
            build_strategy.fuse_elewise_add_act_ops = True
            build_strategy.fuse_bn_act_ops = True
            build_strategy.fuse_bn_add_act_ops = True
            build_strategy.enable_addto = True
        else:
            build_strategy = None
        model = paddle.jit.to_static(model, input_spec=specs, build_strategy=build_strategy)
        logger.info("Successfully to apply @to_static with specs: {}".format(
            specs))

    return model
