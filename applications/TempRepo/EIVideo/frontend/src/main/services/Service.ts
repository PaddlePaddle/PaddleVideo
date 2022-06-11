import { Logger, LoggerFacade } from '/@main/logger'

const emptyLogger: LoggerFacade = {
  log () {},
  warn () {},
  error () {}
}

export const INJECTIONS_SYMBOL = Symbol('__injections__')

export function Inject(type: string) {
  return function (target: any, propertyKey: string) {
    if (!Reflect.has(target, INJECTIONS_SYMBOL)) {
      Reflect.set(target, INJECTIONS_SYMBOL, [])
    }
    if (!type) {
      throw new Error(`Inject recieved type: ${type}!`)
    } else {
      Reflect.get(target, INJECTIONS_SYMBOL).push({ type, field: propertyKey })
    }
  }
}

export class Service {
  readonly name: string
  private logger: LoggerFacade

  constructor(logger: Logger | null) {
    this.name = Object.getPrototypeOf(this).constructor.name
    this.logger = logger == null ? emptyLogger : logger.createLoggerFor(this.name)
  }

  protected log(m: any, ...a: any[]) {
    this.logger.log(`[${this.name}] ${m}`, ...a)
  }

  protected error(m: any, ...a: any[]) {
    this.logger.error(`[${this.name}] ${m}`, ...a)
  }

  protected warn(m: any, ...a: any[]) {
    this.logger.warn(`[${this.name}] ${m}`, ...a)
  }
}
