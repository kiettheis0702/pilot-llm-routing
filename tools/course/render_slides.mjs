/** Import the final PPTX files and render their actual exported contents. */
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const runtime=process.env.CODEX_PRESENTATION_RUNTIME ?? 'C:/Users/ADMIN/.cache/codex-runtimes/codex-primary-runtime/dependencies';
const {PresentationFile,FileBlob}=await import(pathToFileURL(path.join(runtime,'node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs')));
const base=process.argv[2]??'deliverables';
for(const name of ['midterm','poster']){
 const p=await PresentationFile.importPptx(await FileBlob.load(path.resolve(base,name,'8.pptx')));
 const dir=path.resolve('outputs/course-20261030/presentation-final-render',name);await fs.mkdir(dir,{recursive:true});
 for(const [i,s] of p.slides.items.entries()){
  const b=await p.export({slide:s,format:'png',scale:1});await fs.writeFile(path.join(dir,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await b.arrayBuffer()));
 }
 console.log(JSON.stringify({name,rendered:p.slides.items.length}));
}
