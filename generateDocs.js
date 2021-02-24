const path = require('path')
const { PythonShell } = require('python-shell')

const options = {
  mode: 'text',
  pythonPath: 'venv/bin/python',
  pythonOptions: ['-u'],
  // make sure you use an absolute path for scriptPath
  scriptPath: path.resolve(__dirname),
  args: [],
}

PythonShell.run('simple_docstring_parser.py', options, function (err) {
  if (err) throw err
})
