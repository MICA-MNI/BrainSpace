renderWindow_kwds = {'multiSamples': 8, 'lineSmoothing': True,
                     'pointSmoothing': True, 'polygonSmoothing': True}

renderer_kwds = {
    'background': (1, 1, 1)
}

actor_kwds = {
    'specular': .1, 'specularPower': 1, 'diffuse': 1, 'ambient': .05,
    'forceOpaque': True, 'color': (.8, .8, .8)
}


mapper_kwds = {
    'colorMode': 'MapScalars', 'scalarMode': 'UsePointFieldData',
    'useLookupTableScalarRange': True, 'interpolateScalarsBeforeMapping': True
}


scalarBarActor_kwds = {
    'numberOfLabels': 2, 'height': .5, 'position': (.08, .25), 'width': .8,
    'barRatio': .27, 'unconstrainedFontSize': True,
    'annotationLeaderPadding': 2, 'annotationTextScaling': False,
    'FixedAnnotationLeaderLineColor': True,
    'textPad': 2,
    'labelTextProperty': {
        'color': (0, 0, 0), 'italic': False, 'shadow': False, 'bold': True,
        'fontFamily': 'Arial', 'fontSize': 16
    },
    'annotationTextProperty': {
        'color': (0, 0, 0), 'italic': False, 'shadow': False, 'bold': True,
        'fontFamily': 'Arial', 'fontSize': 16
    }
}


textActor_kwds = {
    'textScaleMode': 'viewport',
    'position': (0.5, 0.5),
    'positionCoordinate': {'coordinateSystem': 'normalizedViewport'},
    'position2Coordinate': {'coordinateSystem': 'normalizedViewport'},
    'textProperty': {
        'color': (0, 0, 0), 'italic': False, 'shadow': False, 'bold': True,
        'fontFamily': 'Arial',
        # 'fontSize': 40,
        'fontSize': 50,
        'justification': 'centered',
        'verticalJustification': 'centered'
    }
}


lookuptable_kwds = {
    'nanColor': (0, 0, 0, 1), 'numberOfTableValues': 256
}
