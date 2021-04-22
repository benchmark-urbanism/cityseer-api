const path = require('path')
const { PythonShell } = require('python-shell')

const options = {
  mode: 'text',
  pythonOptions: ['-u'],
  // make sure you use an absolute path for scriptPath
  scriptPath: path.resolve(__dirname),
  args: ['--config', './.yap_config.yaml'],
}
PythonShell.runString('python -m site', null, function (err, result) {
  if (err) throw err
  console.log(result)
})

if (!process.env.CI) {
  options['scriptPath'] = path.resolve(__dirname, '../venv/lib/python3.9/site-packages/yapper')
  options['pythonPath'] = path.resolve(__dirname, '../venv/bin/python')
}

PythonShell.run('__init__.py', options, function (err) {
  if (err) throw err
})
