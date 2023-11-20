DYENV = 'DYLD_LIBRARY_PATH';
olddy = getenv(DYENV);
path_to_add = fullfile(matlabroot, 'bin', 'maci64');
if isempty(olddy)
    newpath = path_to_add;
else
    newpath = [olddy, ':', path_to_add];
end
setenv(DYENV, newpath)