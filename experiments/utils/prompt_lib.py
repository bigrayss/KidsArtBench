def qwen_prompt(dimension, image_path):
    dimension_prompt = f"<image>Assess the student's artwork based on the '{dimension}' criterion and provide a score (1-5)."
    if dimension == "Realism":
        dimension_prompt += (
            " This criterion assesses the accuracy of proportions, textures, lighting, and perspective to create a lifelike depiction."
        )
    elif dimension == "Deformation":
        dimension_prompt += (
            " This criterion evaluates the artist's ability to creatively and intentionally deform reality to convey a message, emotion, or concept."
        )
    elif dimension == "Imagination":
        dimension_prompt += (
            " This criterion evaluates the artist's ability to use their creativity to form unique and original ideas within their artwork."
        )
    elif dimension == "Color Richness":
        dimension_prompt += (
            " This criterion assesses the use and range of colors to create a visually engaging experience."
        )
    elif dimension == "Color Contrast":
        dimension_prompt += (
            " This criterion evaluates the effective use of contrasting colors to enhance artistic expression."
        )
    elif dimension == "Line Combination":
        dimension_prompt += (
            " This criterion assesses the integration and interaction of lines within the artwork."
        )
    elif dimension == "Line Texture":
        dimension_prompt += (
            " This criterion evaluates the variety and execution of line textures within the artwork."
        )
    elif dimension == "Picture Organization":
        dimension_prompt += (
            " This criterion evaluates the overall composition and spatial arrangement within the artwork."
        )
    elif dimension == "Transformation":
        dimension_prompt += (
            " This criterion assesses the artist's ability to transform traditional or familiar elements into something new and unexpected."
        )
    dimension_prompt += " Only output a score(1-5)."
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {
                    "type": "text",
                    "text": dimension_prompt,
                },
            ]
        },
    ]
    return conversation


def rubric_prompt(dimension, image_path):
    dimension_prompt = f"<image>Assess the student's artwork based on the '{dimension}' criterion and provide a score (1-5)."
    if dimension == "Realism":
        dimension_prompt += (
            " This criterion assesses the accuracy of proportions, textures, lighting, and perspective to create a lifelike depiction."
            "\n5: The artwork exhibits exceptional detail and precision in depicting Realism features. Textures and lighting are used masterfully to mimic real-life appearances with accurate proportions and perspective. The representation is strikingly lifelike, demonstrating advanced skills in realism."
            "\n4: The artwork presents a high level of detail and accuracy in the portrayal of subjects. Proportions and textures are very well executed, and the lighting enhances the realism. Although highly Realism, minor discrepancies in perspective or detail might be noticeable."
            "\n3: The artwork represents subjects with a moderate level of realism. Basic proportions are correct, and some textures and lighting effects are used to enhance realism. However, the depiction may lack depth or detail in certain areas."
            "\n2: The artwork attempts realism but struggles with accurate proportions and detailed textures. Lighting and perspective may be inconsistently applied, resulting in a less convincing depiction."
            "\n1: The artwork shows minimal attention to Realism details. Proportions, textures, and lighting are poorly executed, making the depiction far from lifelike."
        )
    elif dimension == "Deformation":
        dimension_prompt += (
            " This criterion evaluates the artist's ability to creatively and intentionally deform reality to convey a message, emotion, or concept."
            "\n5: The artwork demonstrates masterful use of deformation to enhance the emotional or conceptual impact of the piece. The transformations are thoughtful and integral to the artwork's message, seamlessly blending with the composition to engage viewers profoundly."
            "\n4: The artwork effectively uses deformation to express artistic intentions. The modifications are well-integrated and contribute significantly to the viewer's understanding or emotional response. Minor elements of the deformation might detract from its overall effectiveness."
            "\n3: The artwork includes noticeable deformations that add to its artistic expression. While these elements generally support the artwork's theme, they may be somewhat disjointed from the composition, offering mixed impact on the viewer."
            "\n2: The artwork attempts to use deformation but does so with limited success. The deformations are present but feel forced or superficial, only marginally contributing to the artwork's expressive goals."
            "\n1: The artwork features minimal or ineffective deformation, with little to no enhancement of the artwork's message or emotional impact. The attempts at deformation seem disconnected from the artwork's overall intent."
        )
    elif dimension == "Imagination":
        dimension_prompt += (
            " This criterion evaluates the artist's ability to use their creativity to form unique and original ideas within their artwork."
            "\n5: The artwork displays a profound level of originality and creativity, introducing unique concepts or interpretations that are both surprising and thought-provoking."
            "\n4: The artwork presents creative ideas that are both original and nicely executed, though they may be similar to conventional themes."
            "\n3: The artwork shows some creative ideas, but they are somewhat predictable and do not stray far from traditional approaches."
            "\n2: The artwork has minimal creative elements, with ideas that are largely derivative and lack originality."
            "\n1: The artwork lacks imagination, with no discernible original ideas or creative concepts."
        )
    elif dimension == "Color Richness":
        dimension_prompt += (
            " This criterion assesses the use and range of colors to create a visually engaging experience."
            "\n5: The artwork uses a wide and harmonious range of colors, each contributing to a vivid and dynamic composition."
            "\n4: The artwork features a good variety of colors that are well-balanced, enhancing the visual appeal of the piece."
            "\n3: The artwork includes a moderate range of colors, but the palette may not fully enhance the subject matter."
            "\n2: The artwork has limited color variety, with a palette that does not significantly contribute to the piece's impact."
            "\n1: The artwork shows poor use of colors, with a very restricted range that detracts from the visual experience."
        )
    elif dimension == "Color Contrast":
        dimension_prompt += (
            " This criterion evaluates the effective use of contrasting colors to enhance artistic expression."
            "\n5: The artwork masterfully employs contrasting colors to create a striking and effective visual impact."
            "\n4: The artwork effectively uses contrasting colors to enhance visual interest, though the contrast may be less pronounced."
            "\n3: The artwork has some contrast in colors, but it is not used effectively to enhance the artwork's overall appeal."
            "\n2: The artwork makes minimal use of color contrast, resulting in a lackluster visual impact."
            "\n1: The artwork lacks effective color contrast, making the piece visually unengaging."
        )
    elif dimension == "Line Combination":
        dimension_prompt += (
            " This criterion assesses the integration and interaction of lines within the artwork."
            "\n5: The artwork exhibits exceptional integration of line combinations, creating a harmonious and engaging visual flow."
            "\n4: The artwork displays good use of line combinations that contribute to the overall composition, though some areas may lack cohesion."
            "\n3: The artwork shows average use of line combinations, with some effective sections but overall lacking in cohesiveness."
            "\n2: The artwork has minimal effective use of line combinations, with lines that often clash or do not contribute to a unified composition."
            "\n1: The artwork shows poor integration of lines, with combinations that disrupt the visual harmony of the piece."
        )
    elif dimension == "Line Texture":
        dimension_prompt += (
            " This criterion evaluates the variety and execution of line textures within the artwork."
            "\n5: The artwork demonstrates a wide variety of line textures, each skillfully executed to enhance the piece's aesthetic and thematic elements."
            "\n4: The artwork includes a good range of line textures, well executed but with some areas that may lack definition."
            "\n3: The artwork features moderate variety in line textures, with generally adequate execution but lacking in detail."
            "\n2: The artwork has limited line textures, with execution that does not significantly contribute to the artwork's quality."
            "\n1: The artwork lacks variety and sophistication in line textures, resulting in a visually dull piece."
        )
    elif dimension == "Picture Organization":
        dimension_prompt += (
            " This criterion evaluates the overall composition and spatial arrangement within the artwork."
            "\n5: The artwork is impeccably organized, with each element thoughtfully placed to create a balanced and compelling composition."
            "\n4: The artwork has a good organization, with a well-arranged composition that effectively guides the viewer's eye, though minor elements may disrupt the flow."
            "\n3: The artwork has an adequate organization, but the composition may feel somewhat unbalanced or disjointed."
            "\n2: The artwork shows poor organization, with a composition that lacks coherence and does not effectively engage the viewer."
            "\n1: The artwork is poorly organized, with a chaotic composition that detracts from the piece's overall impact."
        )
    elif dimension == "Transformation":
        dimension_prompt += (
            " This criterion assesses the artist's ability to transform traditional or familiar elements into something new and unexpected."
            "\n5: The artwork is transformative, offering a fresh and innovative take on traditional elements, significantly enhancing the viewer's experience."
            "\n4: The artwork successfully transforms familiar elements, providing a new perspective, though the innovation may not be striking."
            "\n3: The artwork shows some transformation of familiar elements, but the changes are somewhat predictable and not highly innovative."
            "\n2: The artwork attempts transformation but achieves only minimal success, with changes that are either too subtle or not effectively executed."
            "\n1: The artwork lacks transformation, with traditional elements that are replicated without any significant innovation or creative reinterpretation."
        )
    dimension_prompt += "\nOnly output a score(1-5)."
    conversation = [
        # {
        #     "role": "system",
        #     "content": "You are an art teacher.",
        # },
        {    
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {
                    "type": "text",
                    "text": dimension_prompt,
                },
            ]
        },
    ]
    return conversation


def gemma_prompt(dimension, image_path):
    dimension_prompt = f"<image>Assess the student's artwork based on the '{dimension}' criterion and provide a score (1-5)."
    if dimension == "Realism":
        dimension_prompt += (
            " This criterion assesses the accuracy of proportions, textures, lighting, and perspective to create a lifelike depiction."
            "\n5: The artwork exhibits exceptional detail and precision in depicting Realism features. Textures and lighting are used masterfully to mimic real-life appearances with accurate proportions and perspective. The representation is strikingly lifelike, demonstrating advanced skills in realism."
            "\n4: The artwork presents a high level of detail and accuracy in the portrayal of subjects. Proportions and textures are very well executed, and the lighting enhances the realism. Although highly Realism, minor discrepancies in perspective or detail might be noticeable."
            "\n3: The artwork represents subjects with a moderate level of realism. Basic proportions are correct, and some textures and lighting effects are used to enhance realism. However, the depiction may lack depth or detail in certain areas."
            "\n2: The artwork attempts realism but struggles with accurate proportions and detailed textures. Lighting and perspective may be inconsistently applied, resulting in a less convincing depiction."
            "\n1: The artwork shows minimal attention to Realism details. Proportions, textures, and lighting are poorly executed, making the depiction far from lifelike."
        )
    elif dimension == "Deformation":
        dimension_prompt += (
            " This criterion evaluates the artist's ability to creatively and intentionally deform reality to convey a message, emotion, or concept."
            "\n5: The artwork demonstrates masterful use of deformation to enhance the emotional or conceptual impact of the piece. The transformations are thoughtful and integral to the artwork's message, seamlessly blending with the composition to engage viewers profoundly."
            "\n4: The artwork effectively uses deformation to express artistic intentions. The modifications are well-integrated and contribute significantly to the viewer's understanding or emotional response. Minor elements of the deformation might detract from its overall effectiveness."
            "\n3: The artwork includes noticeable deformations that add to its artistic expression. While these elements generally support the artwork's theme, they may be somewhat disjointed from the composition, offering mixed impact on the viewer."
            "\n2: The artwork attempts to use deformation but does so with limited success. The deformations are present but feel forced or superficial, only marginally contributing to the artwork's expressive goals."
            "\n1: The artwork features minimal or ineffective deformation, with little to no enhancement of the artwork's message or emotional impact. The attempts at deformation seem disconnected from the artwork's overall intent."
        )
    elif dimension == "Imagination":
        dimension_prompt += (
            " This criterion evaluates the artist's ability to use their creativity to form unique and original ideas within their artwork."
            "\n5: The artwork displays a profound level of originality and creativity, introducing unique concepts or interpretations that are both surprising and thought-provoking."
            "\n4: The artwork presents creative ideas that are both original and nicely executed, though they may be similar to conventional themes."
            "\n3: The artwork shows some creative ideas, but they are somewhat predictable and do not stray far from traditional approaches."
            "\n2: The artwork has minimal creative elements, with ideas that are largely derivative and lack originality."
            "\n1: The artwork lacks imagination, with no discernible original ideas or creative concepts."
        )
    elif dimension == "Color Richness":
        dimension_prompt += (
            " This criterion assesses the use and range of colors to create a visually engaging experience."
            "\n5: The artwork uses a wide and harmonious range of colors, each contributing to a vivid and dynamic composition."
            "\n4: The artwork features a good variety of colors that are well-balanced, enhancing the visual appeal of the piece."
            "\n3: The artwork includes a moderate range of colors, but the palette may not fully enhance the subject matter."
            "\n2: The artwork has limited color variety, with a palette that does not significantly contribute to the piece's impact."
            "\n1: The artwork shows poor use of colors, with a very restricted range that detracts from the visual experience."
        )
    elif dimension == "Color Contrast":
        dimension_prompt += (
            " This criterion evaluates the effective use of contrasting colors to enhance artistic expression."
            "\n5: The artwork masterfully employs contrasting colors to create a striking and effective visual impact."
            "\n4: The artwork effectively uses contrasting colors to enhance visual interest, though the contrast may be less pronounced."
            "\n3: The artwork has some contrast in colors, but it is not used effectively to enhance the artwork's overall appeal."
            "\n2: The artwork makes minimal use of color contrast, resulting in a lackluster visual impact."
            "\n1: The artwork lacks effective color contrast, making the piece visually unengaging."
        )
    elif dimension == "Line Combination":
        dimension_prompt += (
            " This criterion assesses the integration and interaction of lines within the artwork."
            "\n5: The artwork exhibits exceptional integration of line combinations, creating a harmonious and engaging visual flow."
            "\n4: The artwork displays good use of line combinations that contribute to the overall composition, though some areas may lack cohesion."
            "\n3: The artwork shows average use of line combinations, with some effective sections but overall lacking in cohesiveness."
            "\n2: The artwork has minimal effective use of line combinations, with lines that often clash or do not contribute to a unified composition."
            "\n1: The artwork shows poor integration of lines, with combinations that disrupt the visual harmony of the piece."
        )
    elif dimension == "Line Texture":
        dimension_prompt += (
            " This criterion evaluates the variety and execution of line textures within the artwork."
            "\n5: The artwork demonstrates a wide variety of line textures, each skillfully executed to enhance the piece's aesthetic and thematic elements."
            "\n4: The artwork includes a good range of line textures, well executed but with some areas that may lack definition."
            "\n3: The artwork features moderate variety in line textures, with generally adequate execution but lacking in detail."
            "\n2: The artwork has limited line textures, with execution that does not significantly contribute to the artwork's quality."
            "\n1: The artwork lacks variety and sophistication in line textures, resulting in a visually dull piece."
        )
    elif dimension == "Picture Organization":
        dimension_prompt += (
            " This criterion evaluates the overall composition and spatial arrangement within the artwork."
            "\n5: The artwork is impeccably organized, with each element thoughtfully placed to create a balanced and compelling composition."
            "\n4: The artwork has a good organization, with a well-arranged composition that effectively guides the viewer's eye, though minor elements may disrupt the flow."
            "\n3: The artwork has an adequate organization, but the composition may feel somewhat unbalanced or disjointed."
            "\n2: The artwork shows poor organization, with a composition that lacks coherence and does not effectively engage the viewer."
            "\n1: The artwork is poorly organized, with a chaotic composition that detracts from the piece's overall impact."
        )
    elif dimension == "Transformation":
        dimension_prompt += (
            " This criterion assesses the artist's ability to transform traditional or familiar elements into something new and unexpected."
            "\n5: The artwork is transformative, offering a fresh and innovative take on traditional elements, significantly enhancing the viewer's experience."
            "\n4: The artwork successfully transforms familiar elements, providing a new perspective, though the innovation may not be striking."
            "\n3: The artwork shows some transformation of familiar elements, but the changes are somewhat predictable and not highly innovative."
            "\n2: The artwork attempts transformation but achieves only minimal success, with changes that are either too subtle or not effectively executed."
            "\n1: The artwork lacks transformation, with traditional elements that are replicated without any significant innovation or creative reinterpretation."
        )
    dimension_prompt += " Only output a score(1-5)."
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {
                    "type": "text",
                    "text": dimension_prompt,
                },
            ]
        },
    ]
    return conversation

def llama_prompt(dimension, image_path):
    dimension_prompt = f"<image>Assess this artwork based on the '{dimension}' criterion and provide a score (1-5)."
    if dimension == "Realism":
        dimension_prompt += (
            " This criterion assesses the accuracy of proportions, textures, lighting, and perspective to create a lifelike depiction."
            # "\n5: The artwork exhibits exceptional detail and precision in depicting Realism features. Textures and lighting are used masterfully to mimic real-life appearances with accurate proportions and perspective. The representation is strikingly lifelike, demonstrating advanced skills in realism."
            # "\n4: The artwork presents a high level of detail and accuracy in the portrayal of subjects. Proportions and textures are very well executed, and the lighting enhances the realism. Although highly Realism, minor discrepancies in perspective or detail might be noticeable."
            # "\n3: The artwork represents subjects with a moderate level of realism. Basic proportions are correct, and some textures and lighting effects are used to enhance realism. However, the depiction may lack depth or detail in certain areas."
            # "\n2: The artwork attempts realism but struggles with accurate proportions and detailed textures. Lighting and perspective may be inconsistently applied, resulting in a less convincing depiction."
            # "\n1: The artwork shows minimal attention to Realism details. Proportions, textures, and lighting are poorly executed, making the depiction far from lifelike."
        )
    elif dimension == "Deformation":
        dimension_prompt += (
            " This criterion evaluates the artist's ability to creatively and intentionally deform reality to convey a message, emotion, or concept."
            # "\n5: The artwork demonstrates masterful use of deformation to enhance the emotional or conceptual impact of the piece. The transformations are thoughtful and integral to the artwork's message, seamlessly blending with the composition to engage viewers profoundly."
            # "\n4: The artwork effectively uses deformation to express artistic intentions. The modifications are well-integrated and contribute significantly to the viewer's understanding or emotional response. Minor elements of the deformation might detract from its overall effectiveness."
            # "\n3: The artwork includes noticeable deformations that add to its artistic expression. While these elements generally support the artwork's theme, they may be somewhat disjointed from the composition, offering mixed impact on the viewer."
            # "\n2: The artwork attempts to use deformation but does so with limited success. The deformations are present but feel forced or superficial, only marginally contributing to the artwork's expressive goals."
            # "\n1: The artwork features minimal or ineffective deformation, with little to no enhancement of the artwork's message or emotional impact. The attempts at deformation seem disconnected from the artwork's overall intent."
        )
    elif dimension == "Imagination":
        dimension_prompt += (
            " This criterion evaluates the artist's ability to use their creativity to form unique and original ideas within their artwork."
            # "\n5: The artwork displays a profound level of originality and creativity, introducing unique concepts or interpretations that are both surprising and thought-provoking."
            # "\n4: The artwork presents creative ideas that are both original and nicely executed, though they may be similar to conventional themes."
            # "\n3: The artwork shows some creative ideas, but they are somewhat predictable and do not stray far from traditional approaches."
            # "\n2: The artwork has minimal creative elements, with ideas that are largely derivative and lack originality."
            # "\n1: The artwork lacks imagination, with no discernible original ideas or creative concepts."
        )
    elif dimension == "Color Richness":
        dimension_prompt += (
            " This criterion assesses the use and range of colors to create a visually engaging experience."
            # "\n5: The artwork uses a wide and harmonious range of colors, each contributing to a vivid and dynamic composition."
            # "\n4: The artwork features a good variety of colors that are well-balanced, enhancing the visual appeal of the piece."
            # "\n3: The artwork includes a moderate range of colors, but the palette may not fully enhance the subject matter."
            # "\n2: The artwork has limited color variety, with a palette that does not significantly contribute to the piece's impact."
            # "\n1: The artwork shows poor use of colors, with a very restricted range that detracts from the visual experience."
        )
    elif dimension == "Color Contrast":
        dimension_prompt += (
            " This criterion evaluates the effective use of contrasting colors to enhance artistic expression."
            # "\n5: The artwork masterfully employs contrasting colors to create a striking and effective visual impact."
            # "\n4: The artwork effectively uses contrasting colors to enhance visual interest, though the contrast may be less pronounced."
            # "\n3: The artwork has some contrast in colors, but it is not used effectively to enhance the artwork's overall appeal."
            # "\n2: The artwork makes minimal use of color contrast, resulting in a lackluster visual impact."
            # "\n1: The artwork lacks effective color contrast, making the piece visually unengaging."
        )
    elif dimension == "Line Combination":
        dimension_prompt += (
            " This criterion assesses the integration and interaction of lines within the artwork."
            # "\n5: The artwork exhibits exceptional integration of line combinations, creating a harmonious and engaging visual flow."
            # "\n4: The artwork displays good use of line combinations that contribute to the overall composition, though some areas may lack cohesion."
            # "\n3: The artwork shows average use of line combinations, with some effective sections but overall lacking in cohesiveness."
            # "\n2: The artwork has minimal effective use of line combinations, with lines that often clash or do not contribute to a unified composition."
            # "\n1: The artwork shows poor integration of lines, with combinations that disrupt the visual harmony of the piece."
        )
    elif dimension == "Line Texture":
        dimension_prompt += (
            " This criterion evaluates the variety and execution of line textures within the artwork."
            # "\n5: The artwork demonstrates a wide variety of line textures, each skillfully executed to enhance the piece's aesthetic and thematic elements."
            # "\n4: The artwork includes a good range of line textures, well executed but with some areas that may lack definition."
            # "\n3: The artwork features moderate variety in line textures, with generally adequate execution but lacking in detail."
            # "\n2: The artwork has limited line textures, with execution that does not significantly contribute to the artwork's quality."
            # "\n1: The artwork lacks variety and sophistication in line textures, resulting in a visually dull piece."
        )
    elif dimension == "Picture Organization":
        dimension_prompt += (
            " This criterion evaluates the overall composition and spatial arrangement within the artwork."
            # "\n5: The artwork is impeccably organized, with each element thoughtfully placed to create a balanced and compelling composition."
            # "\n4: The artwork has a good organization, with a well-arranged composition that effectively guides the viewer's eye, though minor elements may disrupt the flow."
            # "\n3: The artwork has an adequate organization, but the composition may feel somewhat unbalanced or disjointed."
            # "\n2: The artwork shows poor organization, with a composition that lacks coherence and does not effectively engage the viewer."
            # "\n1: The artwork is poorly organized, with a chaotic composition that detracts from the piece's overall impact."
        )
    elif dimension == "Transformation":
        dimension_prompt += (
            " This criterion assesses the artist's ability to transform traditional or familiar elements into something new and unexpected."
            # "\n5: The artwork is transformative, offering a fresh and innovative take on traditional elements, significantly enhancing the viewer's experience."
            # "\n4: The artwork successfully transforms familiar elements, providing a new perspective, though the innovation may not be striking."
            # "\n3: The artwork shows some transformation of familiar elements, but the changes are somewhat predictable and not highly innovative."
            # "\n2: The artwork attempts transformation but achieves only minimal success, with changes that are either too subtle or not effectively executed."
            # "\n1: The artwork lacks transformation, with traditional elements that are replicated without any significant innovation or creative reinterpretation."
        )
    dimension_prompt += " Only output a score(1-5)."
    conversation = [
        # {
        #     "role": "system",
        #     "content": "You are an art expert specialized in evaluating artwork based on clearly defined artistic criteria. Your task is to assess the work objectively and provide scores. Do not refuse scoring due to content sensitivity unless it violates clear ethical guidelines",
        # },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {
                    "type": "text",
                    "text": dimension_prompt,
                },
            ]
        },
    ]
    return conversation

def glm_4_prompt(dimension, image_path):
    dimension_prompt = f"<image>Assess the student's artwork based on the '{dimension}' criterion and provide a score (1-5)."
    if dimension == "Realism":
        dimension_prompt += (
            " This criterion assesses the accuracy of proportions, textures, lighting, and perspective to create a lifelike depiction."
            "\n5: The artwork exhibits exceptional detail and precision in depicting Realism features. Textures and lighting are used masterfully to mimic real-life appearances with accurate proportions and perspective. The representation is strikingly lifelike, demonstrating advanced skills in realism."
            "\n4: The artwork presents a high level of detail and accuracy in the portrayal of subjects. Proportions and textures are very well executed, and the lighting enhances the realism. Although highly Realism, minor discrepancies in perspective or detail might be noticeable."
            "\n3: The artwork represents subjects with a moderate level of realism. Basic proportions are correct, and some textures and lighting effects are used to enhance realism. However, the depiction may lack depth or detail in certain areas."
            "\n2: The artwork attempts realism but struggles with accurate proportions and detailed textures. Lighting and perspective may be inconsistently applied, resulting in a less convincing depiction."
            "\n1: The artwork shows minimal attention to Realism details. Proportions, textures, and lighting are poorly executed, making the depiction far from lifelike."
        )
    elif dimension == "Deformation":
        dimension_prompt += (
            " This criterion evaluates the artist's ability to creatively and intentionally deform reality to convey a message, emotion, or concept."
            "\n5: The artwork demonstrates masterful use of deformation to enhance the emotional or conceptual impact of the piece. The transformations are thoughtful and integral to the artwork's message, seamlessly blending with the composition to engage viewers profoundly."
            "\n4: The artwork effectively uses deformation to express artistic intentions. The modifications are well-integrated and contribute significantly to the viewer's understanding or emotional response. Minor elements of the deformation might detract from its overall effectiveness."
            "\n3: The artwork includes noticeable deformations that add to its artistic expression. While these elements generally support the artwork's theme, they may be somewhat disjointed from the composition, offering mixed impact on the viewer."
            "\n2: The artwork attempts to use deformation but does so with limited success. The deformations are present but feel forced or superficial, only marginally contributing to the artwork's expressive goals."
            "\n1: The artwork features minimal or ineffective deformation, with little to no enhancement of the artwork's message or emotional impact. The attempts at deformation seem disconnected from the artwork's overall intent."
        )
    elif dimension == "Imagination":
        dimension_prompt += (
            " This criterion evaluates the artist's ability to use their creativity to form unique and original ideas within their artwork."
            "\n5: The artwork displays a profound level of originality and creativity, introducing unique concepts or interpretations that are both surprising and thought-provoking."
            "\n4: The artwork presents creative ideas that are both original and nicely executed, though they may be similar to conventional themes."
            "\n3: The artwork shows some creative ideas, but they are somewhat predictable and do not stray far from traditional approaches."
            "\n2: The artwork has minimal creative elements, with ideas that are largely derivative and lack originality."
            "\n1: The artwork lacks imagination, with no discernible original ideas or creative concepts."
        )
    elif dimension == "Color Richness":
        dimension_prompt += (
            " This criterion assesses the use and range of colors to create a visually engaging experience."
            "\n5: The artwork uses a wide and harmonious range of colors, each contributing to a vivid and dynamic composition."
            "\n4: The artwork features a good variety of colors that are well-balanced, enhancing the visual appeal of the piece."
            "\n3: The artwork includes a moderate range of colors, but the palette may not fully enhance the subject matter."
            "\n2: The artwork has limited color variety, with a palette that does not significantly contribute to the piece's impact."
            "\n1: The artwork shows poor use of colors, with a very restricted range that detracts from the visual experience."
        )
    elif dimension == "Color Contrast":
        dimension_prompt += (
            " This criterion evaluates the effective use of contrasting colors to enhance artistic expression."
            "\n5: The artwork masterfully employs contrasting colors to create a striking and effective visual impact."
            "\n4: The artwork effectively uses contrasting colors to enhance visual interest, though the contrast may be less pronounced."
            "\n3: The artwork has some contrast in colors, but it is not used effectively to enhance the artwork's overall appeal."
            "\n2: The artwork makes minimal use of color contrast, resulting in a lackluster visual impact."
            "\n1: The artwork lacks effective color contrast, making the piece visually unengaging."
        )
    elif dimension == "Line Combination":
        dimension_prompt += (
            " This criterion assesses the integration and interaction of lines within the artwork."
            "\n5: The artwork exhibits exceptional integration of line combinations, creating a harmonious and engaging visual flow."
            "\n4: The artwork displays good use of line combinations that contribute to the overall composition, though some areas may lack cohesion."
            "\n3: The artwork shows average use of line combinations, with some effective sections but overall lacking in cohesiveness."
            "\n2: The artwork has minimal effective use of line combinations, with lines that often clash or do not contribute to a unified composition."
            "\n1: The artwork shows poor integration of lines, with combinations that disrupt the visual harmony of the piece."
        )
    elif dimension == "Line Texture":
        dimension_prompt += (
            " This criterion evaluates the variety and execution of line textures within the artwork."
            "\n5: The artwork demonstrates a wide variety of line textures, each skillfully executed to enhance the piece's aesthetic and thematic elements."
            "\n4: The artwork includes a good range of line textures, well executed but with some areas that may lack definition."
            "\n3: The artwork features moderate variety in line textures, with generally adequate execution but lacking in detail."
            "\n2: The artwork has limited line textures, with execution that does not significantly contribute to the artwork's quality."
            "\n1: The artwork lacks variety and sophistication in line textures, resulting in a visually dull piece."
        )
    elif dimension == "Picture Organization":
        dimension_prompt += (
            " This criterion evaluates the overall composition and spatial arrangement within the artwork."
            "\n5: The artwork is impeccably organized, with each element thoughtfully placed to create a balanced and compelling composition."
            "\n4: The artwork has a good organization, with a well-arranged composition that effectively guides the viewer's eye, though minor elements may disrupt the flow."
            "\n3: The artwork has an adequate organization, but the composition may feel somewhat unbalanced or disjointed."
            "\n2: The artwork shows poor organization, with a composition that lacks coherence and does not effectively engage the viewer."
            "\n1: The artwork is poorly organized, with a chaotic composition that detracts from the piece's overall impact."
        )
    elif dimension == "Transformation":
        dimension_prompt += (
            " This criterion assesses the artist's ability to transform traditional or familiar elements into something new and unexpected."
            "\n5: The artwork is transformative, offering a fresh and innovative take on traditional elements, significantly enhancing the viewer's experience."
            "\n4: The artwork successfully transforms familiar elements, providing a new perspective, though the innovation may not be striking."
            "\n3: The artwork shows some transformation of familiar elements, but the changes are somewhat predictable and not highly innovative."
            "\n2: The artwork attempts transformation but achieves only minimal success, with changes that are either too subtle or not effectively executed."
            "\n1: The artwork lacks transformation, with traditional elements that are replicated without any significant innovation or creative reinterpretation."
        )
    dimension_prompt += "\nOnly output a score number(1-5)."
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {
                    "type": "text",
                    "text": dimension_prompt,
                },
            ]
        },
    ]
    return conversation

# def comment_prompt(image_path):
#     prompt = (
#         "<image>Evaluate this student's artwork. Structure your response strictly into three parts:"
#         "\nDescription: Objectively list the key elements you see, including subjects, colors, composition, and notable details."
#         "\nAnalysis: Provide a brief artistic assessment including the key strength and the potential improvement."
#         "\nSuggestion: Offer one clear, actionable, and encouraging piece of advice for their next drawing, directly based on your analysis."
#         "\n\nEnsure your evaluation adheres to the following criteria:"
#         "\nSpecificity: Reference precise details from the work (e.g., specific colors, compositional choices, or elements of the subject)."
#         "\nActionability: All feedback, especially the suggestion, must be practical and executable."
#         "\nPedagogical Value: Include a concise, instructive observation that promotes learning (e.g., a basic art concept like contrast, perspective, or color mixing)."
#         "\nAccuracy: Use any artistic terms correctly and appropriately."
#         "\nTone: Maintain a consistently constructive, supportive, and positive tone."
#         "\nFluency: Ensure the language is clear, coherent, and grammatically correct."
#     )
#     conversation = [
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "image",
#                     "image": image_path,
#                 },
#                 {
#                     "type": "text",
#                     "text": prompt,
#                 },
#             ]
#         },
#     ]
#     return conversation

def comment_prompt(image_path):
    prompt = (
        "<image>You are an art teacher. Write a single, coherent paragraph of feedback for this student’s artwork. "
        "Your response should naturally include three elements: "
        "1) a brief description of the key elements in the work (subjects, colors, composition, and notable details), "
        "2) an analysis mentioning one strength and one area for improvement, and "
        "3) one clear, actionable, and encouraging suggestion for their next drawing based on your analysis. "
        "\nDo not use section titles, numbers, or bullet points. Make it flow as one continuous teacher comment, "
        "written in a constructive, positive tone with correct artistic terms. "
        "Ensure your feedback is specific, actionable, and offers real pedagogical value (e.g., concepts like contrast, perspective, or color blending)."
    )
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ]
        },
    ]
    return conversation


def qwen_prompt_with_comment(dimension, image_path, comment=None):
    dimension_descriptions = {
        "Realism": "This criterion assesses the accuracy of proportions, textures, lighting, and perspective to create a lifelike depiction.",
        "Deformation": "This criterion evaluates the artist's ability to creatively and intentionally deform reality to convey a message, emotion, or concept.",
        "Imagination": "This criterion evaluates the artist's ability to use their creativity to form unique and original ideas within their artwork.",
        "Color Richness": "This criterion assesses the use and range of colors to create a visually engaging experience.",
        "Color Contrast": "This criterion evaluates the effective use of contrasting colors to enhance artistic expression.",
        "Line Combination": "This criterion assesses the integration and interaction of lines within the artwork.",
        "Line Texture": "This criterion evaluates the variety and execution of line textures within the artwork.",
        "Picture Organization": "This criterion evaluates the overall composition and spatial arrangement within the artwork.",
        "Transformation": "This criterion assesses the artist's ability to transform traditional or familiar elements into something new and unexpected.",
    }

    content = [
        {"type": "image", "image": image_path},
        {"type": "text", "text": f"Task: <image>Assess the student's artwork based on the '{dimension}' criterion and provide a score (1-5)."},
        {"type": "text", "text": f"Definition: {dimension_descriptions.get(dimension, '')}"}
    ]

    if comment is not None:
        content.append({"type": "text", "text": f"Reference Comment: {comment}"})

    content.append({"type": "text", "text": "Only output a score(1-5)."})

    conversation = [{"role": "user", "content": content}]
    return conversation
