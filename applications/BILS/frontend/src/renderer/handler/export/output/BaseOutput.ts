export default interface BaseOutput {
  output(model: any): Promise<string>;
}
