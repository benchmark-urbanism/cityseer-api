const path = require('path')
const { PythonShell } = require('python-shell')

console.log('boo')
console.log(path.resolve(__dirname, '.'))

const options = {
  mode: 'text',
  pythonPath: 'venv/bin/python',
  pythonOptions: ['-u'],
  // make sure you use an absolute path for scriptPath
  scriptPath: path.resolve(__dirname, '.'),
  args: [],
}

PythonShell.run('docs/pydoc_markdown/pydoc_builder.py', options, function (err) {
  if (err) throw err
  console.log('finished')
})
