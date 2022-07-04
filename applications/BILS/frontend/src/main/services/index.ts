import { ipcMain } from 'electron'
import { Logger } from '../logger'
import { BaseService } from './BaseService'
import { FooService } from './FooService'
import { INJECTIONS_SYMBOL } from './Service'
import { FileService } from './FileService'
import { PrettierService } from './PrettierService'
import { VideoService } from './VideoService'

/**
 * All services definition
 */
export interface Services {
  FooService: FooService,
  BaseService: BaseService,
  FileService: FileService,
  PrettierService: PrettierService,
  VideoService: VideoService
}

let _services!: Services

/**
 * Initialize the services module to serve client (renderer process)
 *
 * @param logger The simple app logger
 */
export function initialize (logger: Logger) {
  _initialize({
    BaseService: new BaseService(logger),
    FooService: new FooService(logger),
    FileService: new FileService(logger),
    PrettierService: new PrettierService(logger),
    VideoService: new VideoService(logger)
  })
}

/**
 * Initialize the services module to serve client (renderer process)
 *
 * @param services The running services for current app
 */
function _initialize (services: Services) {
  if (_services) {
    throw new Error('Should not initialize the services multiple time!')
  }
  _services = services
  for (const serv of Object.values(services)) {
    const injects = Object.getPrototypeOf(serv)[INJECTIONS_SYMBOL] || []
    for (const i of injects) {
      const { type, field } = i
      if (type in services) {
        const success = Reflect.set(serv, field, (services as any)[type])
        if (!success) {
          throw new Error(`Cannot set service ${type} to ${Object.getPrototypeOf(serv)}`)
        }
      } else {
        throw new Error(`Cannot find service named ${type}! Which is required by ${Object.getPrototypeOf(serv).constructor.name}`)
      }
    }
  }
}

export class ServiceNotFoundError extends Error {
  constructor (readonly service: string) {
    super(`Cannot find service named ${service}!`)
  }
}

export class ServiceMethodNotFoundError extends Error {
  constructor (readonly service: string, readonly method: string) {
    super(`Cannot find method named ${method} in service [${service}]!`)
  }
}

ipcMain.handle('service:call', (event, name: string, method: string, ...payloads: any[]) => {
  if (!_services) {
    throw new Error('Cannot call any service until the services are ready!')
  }
  const service = (_services as any)[name]
  if (!service) {
    throw new ServiceNotFoundError(name)
  }
  if (!service[method]) {
    throw new ServiceMethodNotFoundError(name, method)
  }
  return service[method](...payloads)
})
