function printStruct(s, indent, fieldName, fid)
% Recursively print a struct's fields and values to fid (default: stdout).
% If a struct has a 'total' field, it is printed inline with the struct name.
%
% Inputs:
%   s         - struct or value to print
%   indent    - indentation string (default: '')
%   fieldName - display name for s (default: variable name)
%   fid       - file identifier (default: 1 = stdout)

    if nargin < 2, indent    = '';            end
    if nargin < 3, fieldName = inputname(1);  end
    if nargin < 4, fid       = 1;             end

    
    if isstruct(s)
        % Print struct name, with 'total' field shown inline if present
        if isfield(s, 'total') && isnumeric(s.total)
            fprintf(fid,'%s%s: %.3e\n', indent, fieldName, s.total);
        elseif ~isempty(fieldName)
            fprintf(fid,'%s%s:\n', indent, fieldName);
        end
    
        % Recurse into each field, skipping 'total' (already printed)
        fields = fieldnames(s);
        for i = 1:length(fields)
            field = fields{i};
            if strcmp(field, 'total')
                continue;  % already printed inline
            end
            value = s.(field);
            printStruct(value, [indent '  '], field, fid);
        end

    else
        % Leaf value — print directly
        fprintf(fid,'%s%s: ', indent, fieldName);
        printFormatted(s, fid);
    end
end

function printFormatted(value, fid)
    % Print a scalar, vector, string, or fallback for other types
    if isnumeric(value)
        if isscalar(value)
            fprintf(fid,'%.3e\n', value);
        else
            str = sprintf('%.3e ', value);
            fprintf(fid,'[%s]\n', strtrim(str));
        end
    elseif ischar(value) || isstring(value)
        fprintf(fid,'%s\n', value);
    else
        try
            disp(value);
        catch
            fprintf(fid,'<unprintable>\n');
        end
    end
end
