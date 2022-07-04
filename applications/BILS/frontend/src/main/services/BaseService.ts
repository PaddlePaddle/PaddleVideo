import {app} from 'electron'
import {platform} from 'os'
import {Service} from './Service'

export class BaseService extends Service {
  async getBasicInformation() {
    this.log('getBasicInformation is called!')
    const result = {
      platform: platform(),
      version: app.getVersion(),
      root: app.getPath('userData')
    }
    return result
  }
}
