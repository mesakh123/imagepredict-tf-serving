from django import template
register = template.Library()

@register.filter
def extras(List ,i ):
    return List[int(i)]
