import BaseModel from '/@/handler/export/model/BaseModel'

class FFModel extends BaseModel {
  async customGenModel(data: any) {
    return data
  }
}

const ffModel = new FFModel()

export {
  ffModel
}
