function [ Shape ] = parseObjMesh(fullfilename)
%PARSEOBJMESH Summary of this function goes here
%   Detailed explanation goes here

% Read the objects from a Wavefront OBJ file
%
% OBJ=read_wobj(filename);
%
% OBJ struct containing:
%
% OBJ.vertices : Vertices coordinates
% OBJ.vertices_texture: Texture coordinates
% OBJ.vertices_normal : Normal vectors
% OBJ.vertices_point  : Vertice data used for points and lines
% OBJ.material : Parameters from external .MTL file, will contain parameters like
%           newmtl, Ka, Kd, Ks, illum, Ns, map_Ka, map_Kd, map_Ks,
%           example of an entry from the material object:
%       OBJ.material(i).type = newmtl
%       OBJ.material(i).data = 'vase_tex'
% OBJ.objects  : Cell object with all objects in the OBJ file,
%           example of a mesh object:
%       OBJ.objects(i).type='f'
%       OBJ.objects(i).data.vertices: [n x 3 double]
%       OBJ.objects(i).data.texture:  [n x 3 double]
%       OBJ.objects(i).data.normal:   [n x 3 double]
%
% Example,
%   OBJ=read_wobj('examples\example10.obj');
%   FV.vertices=OBJ.vertices;
%   FV.faces=OBJ.objects(3).data.vertices;
%   figure, patch(FV,'facecolor',[1 0 0]); camlight
%
% Function is written by D.Kroon University of Twente (June 2010)

% python version can be found here - http://www.pygame.org/wiki/OBJFileLoader

verbose=false;

if(exist('fullfilename','var')==0)
    [filename, filefolder] = uigetfile('*.obj', 'Read obj-file');
    fullfilename = [filefolder filename];
end
filefolder = fileparts( fullfilename);
if(verbose),disp(['Reading Object file : ' fullfilename]); end


% Read the DI3D OBJ textfile to a cell array
file_words = file2cellarray( fullfilename);
% Remove empty cells, merge lines split by "\" and convert strings with values to double
[ftype, fdata]= fixlines(file_words);

% Vertex data
vertices=[]; nv=0;
faces = [];nf = 0;
vertices_texture=[]; nvt=0;
vertices_point=[]; nvp=0;
vertices_normal=[]; nvn=0;
material=[];

% Surface data
no=0;

% Loop through the Wavefront object file
for iline=1:length(ftype)
    if(mod(iline,10000)==0),
        if(verbose),disp(['Lines processed : ' num2str(iline)]); end
    end
    
    type=ftype{iline}; data=fdata{iline};
    
    % Switch on data type line
    switch(type)
        case{'mtllib'}
            continue;
        case('v') % vertices
            nv=nv+1;
            if(length(data)==3)
                % Reserve block of memory
                if(mod(nv,10000)==1), vertices(nv+1:nv+10001,1:3)=0; end
                % Add to vertices list X Y Z
                vertices(nv,1:3)=data;
            else
                % Reserve block of memory
                if(mod(nv,10000)==1), vertices(nv+1:nv+10001,1:4)=0; end
                % Add to vertices list X Y Z W
                vertices(nv,1:4)=data;
            end
        case('vp')
            continue;
        case('vn')
            continue;
        case('vt')
            continue;
        case('l')
            continue;
        case('f')
            nf = nf+1; if(mod(nf,10000)==1), faces(nf:nf+1000,1:3)=0; end
            array_vertices=[];
            array_texture=[];
            array_normal=[];
            for i=1:length(data);
                switch class(data)
                    case 'cell'
                        tvals=str2double(stringsplit(data{i},'/'));
                    case 'string'
                        tvals=str2double(stringsplit(data,'/'));
                    otherwise
                        tvals=data(i);
                end
                val=tvals(1);
                
                if(val<0), val=val+1+nv; end
                array_vertices(i)=val;
                
            end
            
            faces(nf,:) = array_vertices;
        case{'#','$'}
            % Comment
            tline='  %'; 
            if(iscell(data))
                for i=1:length(data), tline=[tline ' ' data{i}]; end
            else
                tline=[tline data];
            end
            if(verbose), disp(tline); end
        case{''}
        otherwise
            continue;
    end
end
% Add all data to output struct
Shape.vertexPoss=vertices(1:nv,:);
Shape.vertexPoss=Shape.vertexPoss';
Shape.faceVIds=faces(1:nf,:);
Shape.faceVIds = Shape.faceVIds';
Shape.has_material = 0;
end

function twords=stringsplit(tline,tchar)
% Get start and end position of all "words" separated by a char
i=find(tline(2:end-1)==tchar)+1; i_start=[1 i+1]; i_end=[i-1 length(tline)];
% Create a cell array of the words
twords=cell(1,length(i_start)); for j=1:length(i_start), twords{j}=tline(i_start(j):i_end(j)); end
end

function file_words=file2cellarray(filename)
% Open a DI3D OBJ textfile
fid=fopen(filename,'r');
file_text=fread(fid, inf, 'uint8=>char')';
fclose(fid);
file_lines = regexp(file_text, '\n+', 'split');
file_words = regexp(file_lines, '\s+', 'split');
end

function [ftype,fdata]=fixlines(file_words)
ftype=cell(size(file_words));
fdata=cell(size(file_words));

iline=0; jline=0;
while(iline<length(file_words))
    iline=iline+1;
    twords=removeemptycells(file_words{iline});
    if(~isempty(twords))
        % Add next line to current line when line end with '\'
        while(strcmp(twords{end},'\')&&iline<length(file_words))
            iline=iline+1;
            twords(end)=[];
            twords=[twords removeemptycells(file_words{iline})];
        end
        % Values to double
        
        type=twords{1};
        stringdold=true;
        j=0;
        switch(type)
            case{'#','$'}
                for i=2:length(twords)
                    j=j+1; twords{j}=twords{i};                    
                end    
            otherwise    
                for i=2:length(twords)
                    str=twords{i};
                    val=str2double(str);
                    stringd=~isfinite(val);
                    if(stringd)
                        j=j+1; twords{j}=str;
                    else
                        if(stringdold)
                            j=j+1; twords{j}=val;
                        else
                            twords{j}=[twords{j} val];    
                        end
                    end
                    stringdold=stringd;
                end
        end
        twords(j+1:end)=[];
        jline=jline+1;
        ftype{jline}=type;
        if(length(twords)==1), twords=twords{1}; end
        fdata{jline}=twords;
    end
end
ftype(jline+1:end)=[];
fdata(jline+1:end)=[];
end

function b=removeemptycells(a)
j=0; b={};
for i=1:length(a);
    if(~isempty(a{i})),j=j+1; b{j}=a{i}; end;
end
end

