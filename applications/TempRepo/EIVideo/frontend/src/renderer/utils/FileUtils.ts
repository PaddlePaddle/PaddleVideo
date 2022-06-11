import { useService } from '/@/composables'
import { FileItem, FileType } from '/@/store/project'
import { v1 as uuidv4 } from 'uuid'

const { getFileExpandName, getFileName, isFile } = useService('FileService')
export async function genFileObj (path: string | null) {
  const fileItem: FileItem = {
    uuid: uuidv4(),
    path: path,
    name: await getFileName(path as string),
    type: await isFile(path as string) ? FileType.File : FileType.FOLDER,
    suffix: await getFileExpandName(path as string),
    isLoadChildren: false,
    children: []
  }
  return fileItem
}
